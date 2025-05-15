import subprocess
import os
import glob
import random
from pathlib import Path
from subprocess import run
import tarfile
import json
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import signal
import jsonlines
from collections import defaultdict
import multiprocessing
import xml.etree.ElementTree as ET
from subprocess import TimeoutExpired,Popen,PIPE
import psutil


defects4j_dir = "/home/ubuntu/defects4j"
d4j_projects_dir = "/data/ubuntu/ut_gen/d4j_projects"
testing_dir = "/data/ubuntu/ut_gen/testing"
generator='evosuite'


def extract(string):
    string.replace('``` java', '```java')
    string.replace('```Java', '```java')
    string.replace('``` Java', '```java')
    methods = string.split('```java')
    ut = []
    for method in methods:
        if '```' in method:
            java_code = method.split('```')[0]
        else:
            java_code = method
        parts = java_code.split("@Test")
        if "@Test" in java_code:
            parts = java_code.split("@Test")
            for part in parts[1:]:
                test_method = "@Test" + part
                ut.append(remove_extra_braces(test_method.strip()))
        else:
            parts = java_code.split("public void")
            for part in parts[1:]:
                test_method = "@Test\npublic void " + part
                ut.append(remove_extra_braces(test_method.strip()))
    return ut


def remove_extra_braces(java_code):
    chars = java_code.strip().split()
    if chars[-1] == '}' and java_code.count('{') - java_code.count('}')==-1:
        return java_code.strip()[:-1]
    else:
        return java_code

def remove_comments(code):
    in_block_comment = False
    in_string = False
    result = []
    i = 0
    while i < len(code):
        if code[i] == '"' and not in_block_comment:
            in_string = not in_string
            result.append(code[i])
        elif not in_string:
            if in_block_comment:
                if code[i:i+2] == '*/':
                    in_block_comment = False
                    i += 1
            elif code[i:i+2] == '/*':
                in_block_comment = True
                i += 1
            elif code[i:i+2] == '//':
                while i < len(code) and code[i] != '\n':
                    i += 1
            else:
                result.append(code[i])
        else:
            result.append(code[i])
        i += 1
    return ''.join(result)

def set_process_group():
    os.setpgrp()


def worker(chunk_data):
    err_log = []
    examples = []
    data = chunk_data['data']
    cfg = chunk_data['cfg']
    thread_id = str(chunk_data['thread_id'])
    for item_target in tqdm(data):
        generate_test_case = item_target['ut']
        project = item_target['project']
        bug = str(item_target['id'])
        import_lines = []
        for line in item_target['import'].split('\n'):
            if 'jupiter' not in line:
                import_lines.append(line)
        item_target['import'] = '\n'.join(import_lines)
    
        test_tar_dir = f'{d4j_projects_dir}/{project}/{generator}/{bug}/{project}-{bug}b-{generator}.{bug}.tar.bz2'
        tar = tarfile.open(test_tar_dir, "r:bz2")
        output = f'{testing_dir}/{project}/{generator}/{bug}/{thread_id}/'
        if os.path.exists(output):
            shutil.rmtree(output)
        if not os.path.exists(output):
            os.makedirs(output)
        tar.extractall(output)
        tar.close()
    
        os.chdir(output)
        test_file = str(list(Path('.').rglob('*ESTest.java'))[0])
        fold_name = test_file.split("/")[0]
        ori_test_dir = f'{testing_dir}/{project}/{generator}/{bug}/{thread_id}/{test_file}'
        with open(ori_test_dir) as f:
            content = f.read()
        contents = content.split("\n")
        content_lines = []
        package_line = ''
        for i in contents:
            if i.startswith('package '):
                package_line = i.strip().split(';')[0]+';'
                continue
            if '@Test' in i:
                break
            else:
                i=str(i)
                content_lines.append(i)
        tmp_content = [package_line+'\n']+[item_target['import']+'\n']+content_lines
        generate_test_case = str(generate_test_case)
        tmp_content.append(generate_test_case)
        tmp_content.append('}')
        final_test = '\n'.join(tmp_content) 
    
    
        output_file = f"{testing_dir}/{project}/{generator}/{bug}/{thread_id}/{fold_name}"
        compressed_filename = f'{testing_dir}/{project}/{generator}/{bug}/{thread_id}/{project}-{bug}b-{generator}.{bug}.tar.bz2'
        with open(ori_test_dir, "w") as java_file:
            java_file.write(final_test)
        with tarfile.open(compressed_filename, "w:bz2") as tar:
            tar.add(output_file, arcname=os.path.basename(output_file))
    
        result_output = f"/data/ubuntu/ut_gen/output/{cfg.mode}/evo_coverage/{project}/{bug}/{thread_id}"
        if os.path.exists(result_output):
            shutil.rmtree(result_output)
    
        path_to_remove = os.path.join("/tmp", cfg.mode, str(thread_id))
        if os.path.exists(path_to_remove):
            shutil.rmtree(path_to_remove)
    
    
        cmd = f'{defects4j_dir}/framework/bin/run_coverage.pl -p {project} -d {testing_dir}/{project}/{generator}/{bug}/{thread_id} -o {result_output} -v {bug}b -t /tmp/{cfg.mode}/{thread_id}'
        def kill(proc_pid:int):
            process= psutil.Process(proc_pid)
            
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
        def pexec(strcmd:str,n_timeout:int):
            try:
                p=Popen(strcmd,shell=True, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p.communicate(timeout=n_timeout)
                return stdout.decode('utf-8'), stderr.decode('utf-8')
            except TimeoutExpired:
                kill(p.pid)
                return None, None
            except Exception as e:
                print(e)
                return None, None
        stdout, stderr = pexec(cmd, 60)
        if stdout is None and  stderr is None:
            continue
        num_ok = stderr.count('.OK')
        num_fail = stderr.count('FAIL')
        pattern = f'/tmp/{cfg.mode}/{thread_id}/run_coverage.pl_*'
        item = glob.glob(pattern)
        for item in glob.glob(pattern):
            if os.path.isdir(item):
                shutil.rmtree(item) 
                
        item_target['syn_err'] = False
        item_target['comp_err'] = False
        item_target['num_pass'] = False
        item_target['test_fail'] = False
        item_target['lines_info'] = {}
        if num_fail>0:
            item_target['comp_err'] = True 
            item_target['stderr'] = stderr
            item_target['stdout'] = stdout
            examples.append(item_target)
            err_log.append(stderr)
        else:
            coverage_dir = f'{result_output}/coverage'
            if not os.path.exists(coverage_dir):
                continue
            with open(coverage_dir) as f:
                content = f.read()
            contents = content.split("\n")
            metrics = contents[1].split(",")
            if len(metrics)==1:
                item_target['syn_err'] = True
                item_target['stderr'] = stderr
                item_target['stdout'] = stdout
                examples.append(item_target)
            if len(metrics)>1 and ('-' in contents[1]):
                item_target['test_fail'] = True 
                item_target['stderr'] = stderr
                item_target['stdout'] = stdout
                examples.append(item_target)
            if len(metrics)>1 and ('-' not in contents[1]):
                item_target['num_pass'] = True
                item_target['stderr'] = ''
                item_target['stdout'] = ''
                lines_total = float(metrics[4])
                lines_covered = float(metrics[5])
                ratio_line = lines_covered/max(1,lines_total)
                item_target['ratio_line'] = ratio_line

                lines_info = {}
                tree = ET.parse(f'{result_output}/coverage_log/{project}/evosuite/{bug}b.{bug}.xml')
                root = tree.getroot()
                for line in root.findall(".//line"):
                    line_number = line.get('number')
                    hits = line.get('hits')
                    if line_number is not None and hits is not None:
                        lines_info[f'{project}-{bug}-{line_number}'] = int(hits)>0
                item_target['lines_info'] = lines_info
                examples.append(item_target)

    return examples



def add_suffix(code, idx):
    suffix = str(idx)
    result = ""
    
    start = code.find("public void", 0)
    if start == -1:
        return code
    
    result += code[:start]
    start_of_function = start + len("public void")
    start_parenthesis = code.find('(', start_of_function)
    
    if start_parenthesis == -1:
        return code
    
    function_name = code[start_of_function:start_parenthesis].strip()
    
    result += f"public void {function_name}{suffix}"
    result += code[start_parenthesis:]
    
    return result
    
def replace_substring(s, a, b):
    result = []
    i = 0
    while i < len(s):
        if s[i:i+len(a)] == a:
            prev_char = s[i-1] if i > 0 else None
            next_char = s[i+len(a)] if i+len(a) < len(s) else None

            if (prev_char is None or not prev_char.isalnum()) and (next_char is None or not next_char.isalnum()):
                result.append(b)
                i += len(a)
                continue
        
        result.append(s[i])
        i += 1
    
    return ''.join(result)
    
def extract_content(input_str):
    start = input_str.find('{')
    if start == -1:
        return input_str
    open_braces = 0
    end = start
    for i in range(start, len(input_str)):
        if input_str[i] == '{':
            open_braces += 1
        elif input_str[i] == '}':
            open_braces -= 1
            if open_braces == 0:
                end = i
                break
    result = input_str[:end+1]
    return result

def evaluate_coverage(cfg, prompt, idx, iteration_number):
    err_log = []
    data = []
    count=0
    all_number=0
    existing = set()
    
    with jsonlines.open(cfg.output_base_dir + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/prediction.jsonl') as f:
        for obj in f:
            if obj['project']=='Gson' and  obj['id']==8:
                continue
            if obj['prompt'].strip()!=prompt.strip():
                continue
            if obj['project']+str(obj['id'])+obj['method']+obj['prediction'] not in existing:
                existing.add(obj['project']+str(obj['id'])+obj['method']+obj['prediction'])
            else:
                continue
            for ut in extract(obj['prediction']):
                count+=1
                all_number+=ut.count('@Test')
                data.append({'project':obj['project'], 'id':obj['id'], 'ut':ut, 'import':obj['import'], 'method':obj['method']})
    print(count)




    if not os.path.exists(cfg.output_base_dir + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/processed_prediction'+str(idx)+'.jsonl'):
        random.shuffle(data)
        processes = int(multiprocessing.cpu_count()*0.4)
        pool = multiprocessing.Pool(processes)
        chunk_size = len(data) // processes
        chunks = [{'thread_id':i, 'cfg': cfg, 'data':data[i:i+chunk_size]} for i in range(0, len(data), chunk_size)]
        results = pool.map(worker, chunks)
        processed_results = []
        for chunk_results in results:
            for res in chunk_results:
                processed_results.append(res)
        with jsonlines.open(cfg.output_base_dir + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/' + 'processed_prediction'+str(idx)+'.jsonl','w') as f:
            f.write_all(processed_results)
    
    
    syn_err = 0
    comp_err = 0
    test_fail = 0
    num_pass = 0
    success_ut = defaultdict(list)
    imports = defaultdict(list)
    zero_test=0
    project_line_info={'Chart':{'Line':0, 'Branch':0, 'All':0, 'Pass':0}, 
                       'Cli':{'Line':0, 'Branch':0, 'All':0, 'Pass':0}, 
                       'Csv':{'Line':0, 'Branch':0, 'All':0, 'Pass':0}, 
                       'Gson':{'Line':0, 'Branch':0, 'All':0, 'Pass':0}, 
                       'Lang':{'Line':0, 'Branch':0, 'All':0, 'Pass':0}}
    with jsonlines.open(cfg.output_base_dir + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/' + '/processed_prediction'+str(idx)+'.jsonl') as f:
        for item in f:
            if item['syn_err']:
                syn_err+=1
                project_line_info[item['project']]['All']+=1
            if item['comp_err']:
                comp_err+=1
                project_line_info[item['project']]['All']+=1
            if item['test_fail']:
                test_fail+=1
                project_line_info[item['project']]['All']+=1
            if 'ratio_line' in item and item['ratio_line']==0:
                zero_test+=1
            if item['num_pass']:
                project_line_info[item['project']]['All']+=1
                project_line_info[item['project']]['Pass']+=1
                num_pass+=item['ut'].count('@Test')
                if item['ut'].strip()[-1]!='}':
                    item['ut'] = extract_content(item['ut'])
                item['ut'] = remove_comments(item['ut'])
                if ' class ' in item['ut'] and len(item['ut'].split(' class ')[1].split('{')[0].strip().split())==1:
                    class_name = item['ut'].split(' class ')[1].split('{')[0].strip()
                    item['ut'] = replace_substring(item['ut'], class_name, class_name+str(len(success_ut[item['project']+'--'+str(item['id'])])))
                elif '\nclass ' in item['ut'] and len(item['ut'].split('\nclass ')[1].split('{')[0].strip().split())==1:
                    class_name = item['ut'].split('\nclass ')[1].split('{')[0].strip()
                    item['ut'] = replace_substring(item['ut'], class_name, class_name+str(len(success_ut[item['project']+'--'+str(item['id'])])))
                success_ut[item['project']+'--'+str(item['id'])].append(add_suffix(item['ut'], len(success_ut[item['project']+'--'+str(item['id'])])))
                imports[item['project']+'--'+str(item['id'])].append(item['import'])
    print('Zero Test:', zero_test)
    processed_data = []
    for bug in success_ut:
        processed_data.append({'project':bug.split('--')[0], 'id':bug.split('--')[1], 'ut':'\n'.join(success_ut[bug]), 'import':'\n'.join(imports[bug]) })



    all_num_project_bug = 1
    all_line_ratio = 0
    all_branch_ratio = 0
    examples = []
    processed_data = sorted(processed_data, key=lambda x: (x['project'], x['id']))
    lines_info = {}
    with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
        f.write('\n'+prompt+'\n')
    for item_target in processed_data:
        generate_test_case = item_target['ut']
        project = item_target['project']
        bug = str(item_target['id'])
        all_num_project_bug += 1
    
        test_tar_dir = f'{d4j_projects_dir}/{project}/{generator}/{bug}/{project}-{bug}b-{generator}.{bug}.tar.bz2'
        tar = tarfile.open(test_tar_dir, "r:bz2")
        output = f'{testing_dir}/{project}/{generator}/{bug}/final/'
        if not os.path.exists(output):
            os.makedirs(output)
        tar.extractall(output)
        tar.close()
    
        os.chdir(output)
        test_file = str(list(Path('.').rglob('*ESTest.java'))[0])
        fold_name = test_file.split("/")[0]
        ori_test_dir = f'{testing_dir}/{project}/{generator}/{bug}/final/{test_file}'
        with open(ori_test_dir) as f:
            content = f.read()
        contents = content.split("\n")
        content_lines = []
        package_line = ''
        for i in contents:
            if i.startswith('package '):
                package_line = i.strip().split(';')[0]+';'
                continue
            if '@Test' in i:
                break
            else:
                i = str(i)
                content_lines.append(i)
        tmp_content = [package_line+'\n']+[item_target['import']+'\n']+content_lines
        generate_test_case = str(generate_test_case)
        tmp_content.append(generate_test_case)
        tmp_content.append('}')
        final_test = '\n'.join(tmp_content)
            
        output_file = f"{testing_dir}/{project}/{generator}/{bug}/final/{fold_name}"
        compressed_filename = f'{testing_dir}/{project}/{generator}/{bug}/final/{project}-{bug}b-{generator}.{bug}.tar.bz2'
        with open(ori_test_dir, "w") as java_file:
            java_file.write(final_test)
        with tarfile.open(compressed_filename, "w:bz2") as tar:
            tar.add(output_file, arcname=os.path.basename(output_file))
    
        result_output = f"/data/ubuntu/ut_gen/output/{cfg.mode}/evo_coverage/{project}/{bug}"
        if os.path.exists(result_output):
            shutil.rmtree(result_output)
    
        cmd = f'{defects4j_dir}/framework/bin/run_coverage.pl -p {project} -d {testing_dir}/{project}/{generator}/{bug}/final -o {result_output} -v {bug}b'
        def kill(proc_pid:int):
            process= psutil.Process(proc_pid)
            
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
        def pexec(strcmd:str,n_timeout:int):
            try:
                p=Popen(strcmd,shell=True, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p.communicate(timeout=n_timeout)
                return stdout.decode('utf-8'), stderr.decode('utf-8')
            except TimeoutExpired:
                kill(p.pid)
                return None, None
            except Exception as e:
                print(e)
                return None, None
        stdout, stderr = pexec(cmd, 600)
        if stdout is None and  stderr is None:
            with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
                f.write(str([project, bug, '------------------'])+'\n')
            continue

        num_ok = stderr.count('.OK')
        num_fail = stderr.count('FAIL')
        pattern = f'/tmp/{cfg.mode}/run_coverage.pl_*'
        item = glob.glob(pattern)
        for item in glob.glob(pattern):
            if os.path.isdir(item):
                shutil.rmtree(item)
        if num_fail>0:
            with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
                f.write(str(final_test)+'\n')
                f.write(str(stderr)+'\n')
            err_log.append(stderr)
            ratio_line = 0.0
            ratio_branch = 0.0
        else:
            coverage_dir = f'{result_output}/coverage'
            if not os.path.exists(coverage_dir):
                continue
            with open(coverage_dir) as f:
                content = f.read()
            contents = content.split("\n")
            metrics = contents[1].split(",")
            with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
                f.write(str(metrics)+'\n')
            if len(metrics)>1 and ('-' not in contents[1]):
                lines_total = float(metrics[4])
                lines_covered = float(metrics[5])
                branches_total = float(metrics[6])
                branches_covered = float(metrics[7])
                ratio_line = lines_covered/max(1,lines_total)
                ratio_branch = branches_covered/max(1,branches_total)
                all_line_ratio += ratio_line
                all_branch_ratio += ratio_branch
                project_line_info[project]['Line']+=ratio_line
                project_line_info[project]['Branch']+=ratio_branch
                item_target['lines_total'] = lines_total
                item_target['lines_covered'] = lines_covered
                item_target['branches_total'] = branches_total
                item_target['branches_covered'] = branches_covered
                item_target['ratio_line'] = ratio_line
                item_target['ratio_branch'] = ratio_branch
                examples.append(item_target)

                tree = ET.parse(f'{result_output}/coverage_log/{project}/evosuite/{bug}b.{bug}.xml')
                root = tree.getroot()
                for line in root.findall(".//line"):
                    line_number = line.get('number')
                    hits = line.get('hits')
                    if line_number is not None and hits is not None:
                        lines_info[f'{project}{bug}{line_number}'] = int(hits)>0
            else:
                with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
                    f.write(str(final_test)+'\n')
                    f.write(str(stderr)+'\n')
                    f.write(str(stdout)+'\n')


    if cfg.stage=='train':
        avg_line_ratio = all_line_ratio/10
        avg_branch_ratio = all_branch_ratio/10
    else:
        avg_line_ratio = all_line_ratio/147
        avg_branch_ratio = all_branch_ratio/147
    with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
        f.write(f'all_num_tests: {all_number}, Syntactic error: {syn_err}, Comp error: {comp_err}, Fail test case: {test_fail}, all_num_pass: {num_pass}, avg_line_ratio: {round(avg_line_ratio*100, 2)} avg_branch_ratio: {round(avg_branch_ratio*100, 2)}')
    
    with open(cfg.output_base_dir+cfg.mode+'/'+cfg.stage+'/'+str(iteration_number)+'/'+'results'+str(idx)+'.txt','a') as f:
        for project in project_line_info:
            if project == 'Chart':
                print(project, round(project_line_info[project]["Line"]*100/26, 2), round(project_line_info[project]["Branch"]*100/26, 2), round(project_line_info[project]["Pass"]*100/project_line_info[project]["All"], 2))
                f.write(f'{project}  {round(project_line_info[project]["Line"]*100/26, 2)} {round(project_line_info[project]["Branch"]*100/26, 2)} \n')
            elif project == 'Cli':
                print(project, round(project_line_info[project]["Line"]*100/29, 2), round(project_line_info[project]["Branch"]*100/29, 2), round(project_line_info[project]["Pass"]*100/project_line_info[project]["All"], 2))
                f.write(f'{project}  {round(project_line_info[project]["Line"]*100/29, 2)} {round(project_line_info[project]["Branch"]*100/29, 2)} \n')
            elif project == 'Csv':
                print(project, round(project_line_info[project]["Line"]*100/15, 2), round(project_line_info[project]["Branch"]*100/15, 2), round(project_line_info[project]["Pass"]*100/project_line_info[project]["All"], 2))
                f.write(f'{project}  {round(project_line_info[project]["Line"]*100/15, 2)} {round(project_line_info[project]["Branch"]*100/15, 2)} \n')
            elif project == 'Gson':
                print(project, round(project_line_info[project]["Line"]*100/17, 2), round(project_line_info[project]["Branch"]*100/17, 2), round(project_line_info[project]["Pass"]*100/project_line_info[project]["All"], 2))
                f.write(f'{project}  {round(project_line_info[project]["Line"]*100/17, 2)} {round(project_line_info[project]["Branch"]*100/17, 2)} \n')
            elif project == 'Lang':
                print(project, round(project_line_info[project]["Line"]*100/60, 2), round(project_line_info[project]["Branch"]*100/60, 2), round(project_line_info[project]["Pass"]*100/project_line_info[project]["All"], 2))
                f.write(f'{project}  {round(project_line_info[project]["Line"]*100/60, 2)} {round(project_line_info[project]["Branch"]*100/60, 2)} \n')
    print('avg_line_ratio: ', round(avg_line_ratio*100, 2), 'avg_branch_ratio: ', round(avg_branch_ratio*100, 2), flush=True)
    return round(avg_line_ratio*100, 2), round(avg_branch_ratio*100, 2), lines_info




