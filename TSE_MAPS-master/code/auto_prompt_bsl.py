import os
import re
import sys
import regex
import openai
import shutil
import random
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm
from time import sleep
from collections import defaultdict
import numpy as np
import jsonlines
import editdistance
import argparse
from evaluate import evaluate_coverage, evaluate_detection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
import tarfile
import jsonlines
import xml.etree.ElementTree as ET


gpt_api_keys = []
deepinfra_api_keys = []
llm_api_key = []

input_output_pair = '''Input:
AbstractCategoryItemRenderer extends AbstractRenderer implements CategoryItemRenderer, Cloneable, PublicCloneable, Serializable
{
    public CategoryURLGenerator getURLGenerator(int row, int column, boolean selected)
    {
        CategoryURLGenerator generator = (CategoryURLGenerator) this.urlGeneratorList.get(row);
        if (generator == null)
        {
            generator = this.baseURLGenerator;
        }
        return generator;
    }
}
Output:
@Test
public void testgetURLGenerator1() {
    CategoryURLGenerator generator = new CategoryURLGenerator(1,2,True);
    assertNotNull(generator);
}

Input:
FastDateParser implements DateParser, Serializable { @Override public Date parse(String source) throws ParseException { Date date= parse(source, new ParsePosition(0)); if(date==null) { if (locale.equals(JAPANESE_IMPERIAL)) { throw new ParseException( \"(The \" +locale + \" locale does not support dates before 1868 AD)\\n\" + \"Unparseable date: \\\"\"+source+\"\\\" does not match \"+parsePattern.pattern(), 0); } throw new ParseException(\"Unparseable date: \\\"\"+source+\"\\\" does not match \"+parsePattern.pattern(), 0); } return date; } }
Output:
public class FastDateParserTest {
    private FastDateParser parser;
    private SimpleDateFormat dateFormat;
    @Before
    public void setUp() {
        parser = new FastDateParser("yyyy-MM-dd", Locale.US);
        dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.US);
    }
    @Test
    public void testParse() {
        String validDate = "2024-05-14";
        try {
            Date expectedDate = dateFormat.parse(validDate);
            Date parsedDate = parser.parse(validDate);
            assertEquals(expectedDate, parsedDate);
        } catch (ParseException e) {
            fail("ParseException was not expected for a valid date");
        }
    }
}

Input:
TimeSeries extends Series implements Cloneable, Serializable { public Number getValue(RegularTimePeriod period) { int index = getIndex(period); if (index >= 0) { return getValue(index); } else { return null; } } }
Output:
public class TimeSeriesTest {
    @Test
    public void testAddAndGetValue() {
        TimeSeries timeSeries = new TimeSeries("Test Series");
        RegularTimePeriod period = new Day(1, 1, 2020);
        timeSeries.add(period, 100.0);
        Number value = timeSeries.getValue(period);
        assertNotNull(value);
        assertEquals(100.0, value.doubleValue(), 0.001);
    }
 }'''
error_file = open("stderr.txt", "wb")




def post_process(string, idx):
    string.replace('``` java', '```java')
    string.replace('```Java', '```java')
    string.replace('``` Java', '```java')
    methods = string.split('```java')
    ut = []
    import_statements = ''
    for method in methods:
        if '```' in method:
            if 'import' not in method.strip():
                java_code = method.split('```')[0]
                parts = java_code.split("@Test")
                for part in parts[1:]:
                    test_method = "@Test" + part
                    ut.append(test_method.strip())
            else:
                lines = method.split('```')[0].splitlines()
                filtered_lines = [line for line in lines if line.strip().startswith('import ') and 'jupiter' not in line]
                import_statements = import_statements+('\n'.join(filtered_lines))+'\n'
                other_lines = '/n'.join([line for line in lines if (not line.strip().startswith('import '))])
                if len(other_lines.strip())>0:
                    ut.append(other_lines)
    return ut, import_statements


def gpt_official(chunk_data):
    cfg = chunk_data['cfg']
    data = chunk_data['data']
    llm_api_key = chunk_data['api_key']
    openai.api_key = llm_api_key
    output_path = chunk_data['output_path']
    proc_id = chunk_data['proc_id']

    existing_prediction = set()
    if os.path.exists(os.path.join(output_path, f'prediction{proc_id}.jsonl')):
        with jsonlines.open(os.path.join(output_path, f'prediction{proc_id}.jsonl')) as file:
            for item in file:
                existing_prediction.add(item['prompt'].strip()+'\n'+item['method'].strip())

    for idx in tqdm(range(len(data))):
        prompt = data[idx]['prompt'].strip()+'\n'+data[idx]['method'].strip()
        if prompt in existing_prediction:
            continue
        success = 0
        fail_count = 0
        while success!=1:
            messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to write unit test cases. Please follow the instructions and make each test case in one individual code block (```java ```). Please make sure that each individual test case is complete. Please reply with {} test cases.".format(cfg.max_test_cases)},
                    {"role": "user", "content": "prompt:"+data[idx]['prompt']+"method:"+data[idx]['method']},
                    ]
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, n=1, temperature=0)
                success=1
                ut, import_statements = post_process(response["choices"][0]['message']['content'].strip(), idx)
                answer = {'prompt':data[idx]['prompt'], 'projects':data[idx]['projects'], 'method':data[idx]['method'], 'ut':ut, 'import':import_statements, 'prediction':response["choices"][0]['message']['content'].strip()}
                with jsonlines.open(os.path.join(output_path, f'prediction{proc_id}.jsonl'), 'a') as f:
                        f.write_all([answer])
                sleep(0.5)
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>10:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break


def llama_official(chunk_data):
    cfg = chunk_data['cfg']
    data = chunk_data['data']
    llm_api_key = chunk_data['api_key']
    openai.api_key = llm_api_key
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    output_path = chunk_data['output_path']
    proc_id = chunk_data['proc_id']

    existing_prediction = set()
    if os.path.exists(os.path.join(output_path, f'prediction{proc_id}.jsonl')):
        with jsonlines.open(os.path.join(output_path, f'prediction{proc_id}.jsonl')) as file:
            for item in file:
                existing_prediction.add(item['prompt'].strip()+'\n'+item['method'].strip())

    for idx in tqdm(range(len(data))):
        prompt = data[idx]['prompt'].strip()+'\n'+data[idx]['method'].strip()
        if prompt in existing_prediction:
            continue
        success = 0
        fail_count = 0
        while success!=1:
            messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to write unit test cases. Please follow the instructions and make each test case in one individual code block (```java ```). Please make sure that each individual test case is complete. Please reply with {} test cases.".format(cfg.max_test_cases)},
                    {"role": "user", "content": "prompt:"+data[idx]['prompt']+"method:"+data[idx]['method']},
                    ]
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, n=1, temperature=0)
                success=1
                ut, import_statements = post_process(response["choices"][0]['message']['content'].strip(), idx)
                answer = {'prompt':data[idx]['prompt'], 'projects':data[idx]['projects'], 'method':data[idx]['method'], 'ut':ut, 'import':import_statements, 'prediction':response["choices"][0]['message']['content'].strip()}
                with jsonlines.open(os.path.join(output_path, f'prediction{proc_id}.jsonl'), 'a') as f:
                        f.write_all([answer])
                sleep(1)
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>10:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break


def ut_generation(cfg, ut_data, prompts, iteration_number):
    output_path = cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number)
    tmp_path = output_path + '/' + 'tmp'
    print('generating test cases...')
    data = []
    if 'gpt' in cfg.model_name:
        api_keys = gpt_api_keys
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        api_keys = deepinfra_api_keys
    pool = multiprocessing.Pool(len(api_keys))
    if len(prompts)!=len(set(prompts)):
        print('redundant')
        input()
    existing_data = []
    if os.path.exists(os.path.join(output_path, 'prediction.jsonl')):
        with jsonlines.open(os.path.join(output_path, 'prediction.jsonl')) as f:
            for obj in f:
                existing_data.append({'prompt':obj['prompt'].strip(), 'method':obj['method'].strip()})
    for prompt in prompts:
        print(prompt)
        for obj in ut_data:
            if obj['projects'] == [{'project':'Gson', 'id':8}]:
                continue
            if {'prompt':prompt[0].strip(), 'method':obj['method'].strip()} in existing_data:
                continue
            data.append({'prompt':prompt[0], 'method':obj['method'], 'projects':obj['projects']})

    chunks_ut_data = np.array_split(data,len(api_keys))
    chunks_data = []
    for i in range(len(api_keys)):
        tmp_data={}
        tmp_data['data'] = chunks_ut_data[i]
        tmp_data['api_key'] = api_keys[i]
        tmp_data['cfg'] = cfg
        tmp_data['output_path'] = tmp_path
        tmp_data['proc_id'] = i
        chunks_data.append(tmp_data)

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    if 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        pool.map(llama_official, chunks_data)
    elif 'gpt' in cfg.model_name:
        pool.map(gpt_official, chunks_data)

    prediction_data = []
    if os.path.exists(os.path.join(output_path, 'prediction.jsonl')):
        with jsonlines.open(os.path.join(output_path, 'prediction.jsonl')) as f:
            for obj in f:
                prediction_data.append(obj)
    with jsonlines.open(os.path.join(output_path, 'prediction.jsonl'), 'w') as f:
        for file in os.listdir(tmp_path):
            file_path = os.path.join(tmp_path, file)
            with jsonlines.open(file_path, 'r') as in_file:
                for obj in in_file:
                    for project in obj['projects']:
                        prediction_data.append({'prompt':obj['prompt'], 'project':project['project'], 'id':project['id'],  'method':obj['method'], 'import':obj['import'], 'ut':obj['ut'], 'prediction':obj['prediction']})
        f.write_all(prediction_data)
    shutil.rmtree(tmp_path)


def error_message_analysis(cfg, files, best_prompt, ut_data, best_prompt_performance, inducted_rules, iteration_number):
    failures = []
    for file in files:
        with jsonlines.open(file) as f:
            for obj in f:
                if 'ratio_line' in obj and obj['ratio_line']==0:
                    err_info = 'low coverage rate'
                    simple_info = 'low coverage rate'
                    failures.append({'project':obj['project'], 'id':obj['id'], 'ut':obj['ut'], 'method':obj['method'], 'err_info':err_info, 'simple_info':simple_info})
                    continue
                if 'BUILD FAILED' not in obj['stderr']:
                    continue
                in_error = False
                simple_info=''
                err_info = ''
                for line in obj['stderr'].split('\n'):
                    if not in_error and line.strip().startswith('[javac]') and 'error' in line:
                        in_error = True
                        err_info += 'error'+line.split('error')[1]+'\n'
                        simple_info = 'error'+line.split('error')[1]
                        continue
                    if in_error and 'warning' not in line and 'error' not in line:
                        err_info+=line.split('[javac] ')[1]+'\n'
                    elif in_error:
                        in_error = False
                        break
                failures.append({'project':obj['project'], 'id':obj['id'], 'ut':obj['ut'], 'method':obj['method'], 'err_info':err_info, 'simple_info':simple_info})
    texts_to_cluster = [obj['simple_info'] for obj in failures]

    # DBSCAN
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts_to_cluster)
    eps = 1
    min_samples = 10
    k=3
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    clusters = defaultdict(list)
    for idx, label in enumerate(dbscan.labels_):
        clusters[label].append(texts_to_cluster[idx])
    
    # weighted sampling
    tried_rules = []
    if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/'+'tried_rules.jsonl'):
        with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/'+'tried_rules.jsonl') as f:
            for obj in f:
                tried_rules.append({'rule': obj['rule'], 'example':obj['example']})
    labels, counts = np.unique(dbscan.labels_, return_counts=True)
    all_clusters = labels[np.argsort(counts)]
    weight = []
    representative_elements = defaultdict(list)
    for cluster in all_clusters:
        cluster_elements = np.where(dbscan.labels_ == cluster)[0]
        centroid = X[cluster_elements].mean(axis=0)
        distances = np.linalg.norm(X[cluster_elements] - centroid, axis=1)
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices[:k]:
            representative_elements[cluster].append(failures[cluster_elements[idx]])
        if len(tried_rules)>0:
            print(failures[cluster_elements[sorted_indices[0]]]['simple_info'])
            sim = []
            for past_selection in tried_rules:
                sim.append(1-editdistance.eval(past_selection['example'], failures[cluster_elements[sorted_indices[0]]]['simple_info'])/(len(past_selection['example'])+len(failures[cluster_elements[sorted_indices[0]]]['simple_info'])))
            weight.append(len(cluster_elements)*(1-1*max(sim)))
        else:
            weight.append(len(cluster_elements))
    print(weight)
    selected_clusters = list(np.random.choice(all_clusters, size=1, replace=False, p=np.array(weight)/sum(weight)))

    # rule induction
    new_inducted_rules = []
    if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/' +'pre_rules.jsonl'):
        with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/' +'pre_rules.jsonl') as f:
            for obj in f:
                print('rule:', obj['rule'])
                print('examples:', obj['example'])
                print('-----------------------------')
                new_inducted_rules.append((obj['rule'], obj['example']))
    else:
        for cluster in selected_clusters:
            selected_failures = representative_elements[cluster]
            if 'gpt' in cfg.model_name:
                openai.api_key = llm_api_key[0]
                model = cfg.model_name
                success = 0
                fail_count = 0
                examples = []
                for idx, obj in enumerate(selected_failures):
                    examples.append('{}. unit test:{}  error information:{}'.format(str(idx), obj['ut'], obj['simple_info']))
                flatten_examples = '\n'.join(examples)
                while success!=1:
                    analysis_messgae =  [
                    {"role": "system", "content": "You are a software engineer and now you will help to analyze the program and give suggestions."},
                    {"role": "user", "content": 'Here are some examples of buggy unit tests along with their error messages. Please identify the causes of these errors and provide a strategy to avoid such errors in the future. Ensure that your recommendation is broadly applicable to similar types of errors, rather than being specific to these particular examples. Additionally, note that suggestions about using IDEs, debugging tools, error messages, and code reviews are not allowed. {}'.format(flatten_examples)}
                    ]
                    try:
                        response = openai.ChatCompletion.create(model=model, messages=analysis_messgae, n=1, temperature=0)
                        solution = response["choices"][0]['message']['content']
                        print('solution:', solution)
                        print('******************************')
                        
                        induction_messgae =  [
                           {"role": "system", "content": "You are a tutor and now you will help to write rules. Directly give the content of the rules. Do not include anything else like \"Rule:\" in your answer. Note that the students are not allowed to debug and they will not know the error information."},
                           {"role": "user", "content": f'Here are some examples of common mistakes students make when writing unit tests and their solutions. Based on these examples, please select one most effective rule and rewrite it into one precise sentence with the format "Ensure that ..." to help these students avoid these mistakes in future unit tests. Ensure that your rule is broadly applicable to similar types of errors, rather than being specific to these particular examples, but you can give examples by adding "such as " in your rule. Note that the students are not allowed to debug or use IDEs, the rule should help them to complete the task by themselves in one go. The suggestion is: {solution}'}
                        ]
                        responses = openai.ChatCompletion.create(model=model, messages=induction_messgae, n=3, temperature=1)
                        for response in responses["choices"]:
                            rule = response['message']['content']
                            new_inducted_rules.append((rule,examples[0].split('error information:')[1]))
                            print('rule:', rule)
                            print('examples:', examples[0].split('error information:')[1])
                            print('-----------------------------')
                        success=1
                    except Exception as e:
                        info = e.args[0]
                        fail_count+=1
                        if 'Max retries exceeded with url:' in info:
                            sleep(2*fail_count)
                        print(info)
                    if fail_count>10:
                        print('{} fail more than 10 times'.format(str(fail_count)))
                        break
            elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
                openai.api_key = llm_api_key[0]
                openai.api_base = "https://api.deepinfra.com/v1/openai"
                model = cfg.model_name
                success = 0
                fail_count = 0
                examples = []
                for idx, obj in enumerate(selected_failures):
                    examples.append('{}. unit test:{}  error information:{}'.format(str(idx), 'Class myTest{\n'+obj['ut']+'\n}', obj['simple_info']))
                flatten_examples = '\n'.join(examples)
                while success!=1:
                    analysis_messgae =  [
                    {"role": "system", "content": "You are a software engineer and now you will help to analyze the program and give suggestions."},
                    {"role": "user", "content": 'Here are some examples of buggy unit tests along with their error messages. Please identify the causes of these errors and provide a strategy to avoid such errors in the future. Ensure that your recommendation is broadly applicable to similar types of errors, rather than being specific to these particular examples. Additionally, note that suggestions about using IDEs, debugging tools, error messages, and code reviews are not allowed. {}'.format(flatten_examples)}
                    ]
                    try:
                        response = openai.ChatCompletion.create(model=model, messages=analysis_messgae, temperature=0)
                        solution = response["choices"][0]['message']['content']
                        print('solution:', solution)
                        print('******************************')
                        
                        induction_messgae =  [
                           {"role": "system", "content": "You are a tutor and now you will help to write rules. Directly give the content of the rules. Do not include anything else like \"Rule:\" in your answer. Note that the students are not allowed to debug and they will not know the error information."},
                           {"role": "user", "content": f'Here are some examples of common mistakes students make when writing unit tests and their solutions. Based on these examples, please select one most effective rule and rewrite it into one precise sentence with the format "Ensure that ..." to help these students avoid these mistakes in future unit tests. Ensure that your rule is broadly applicable to similar types of errors, rather than being specific to these particular examples, but you can give examples by adding "such as " in your rule. Note that the students are not allowed to debug or use IDEs, the rule should help them to complete the task by themselves in one go. The suggestion is: {solution}'}
                        ]
                        success_number=0
                        while success_number!=3:
                            responses = openai.ChatCompletion.create(model=model, messages=induction_messgae, temperature=1)
                            for response in responses["choices"]:
                                rule = response['message']['content']
                                new_inducted_rules.append((rule,examples[0].split('error information:')[1]))
                                print('rule:', rule)
                                print('examples:', examples[0].split('error information:')[1])
                                print('-----------------------------')
                            success_number+=1
                        success=1
                    except Exception as e:
                        info = e.args[0]
                        fail_count+=1
                        if 'Max retries exceeded with url:' in info:
                            sleep(2*fail_count)
                        print(info)
                    if fail_count>10:
                        print('{} fail more than 10 times'.format(str(fail_count)))
                        break
                    
    with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(iteration_number) + '/' +'pre_rules.jsonl', 'w') as f:
        write_data = []
        for obj in new_inducted_rules:
            write_data.append({'rule': obj[0], 'example':obj[1]})
        f.write_all(write_data)
    with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/'+'tried_rules.jsonl', 'a') as f:
        write_data = []
        for obj in new_inducted_rules:
            write_data.append({'rule': obj[0], 'example':obj[1]})
        f.write_all(write_data)
    new_inducted_rules = rule_validation(cfg, new_inducted_rules, inducted_rules, 3, best_prompt, ut_data, best_prompt_performance)
    return inducted_rules+new_inducted_rules



def rule_validation(cfg, rules, existing_rules, number, best_prompt, ut_data, best_prompt_performance):
    print(' rule validation -----------------------------')
    performance = []
    effective_rules = []
    prompts = []
    basic_prompt = best_prompt+' Specifically, please follow the following rules when writing test cases: ' + ' '.join([f'{idx+1}. {rule[0]}' for idx, rule in enumerate(existing_rules)])
    for rule in rules:
        prompts.append((basic_prompt+f' {len(existing_rules)+1}. '+rule[0], 100+len(prompts)))
    ut_generation(cfg, ut_data, prompts, 100+len(existing_rules))

    existing_performance = {}
    if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(100+len(existing_rules)) + '/' +'performance.jsonl'):
        with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(100+len(existing_rules)) + '/' +'performance.jsonl') as f:
            for item in f:
                existing_performance[item['prompt'].strip()] = item
    for idx in range(len(prompts)):
        print('prompt: {}'.format(prompts[idx][0].strip()))
        if prompts[idx][0].strip() in existing_performance:
            print('average line coverage: {}, average condition coverage: {}'.format(existing_performance[prompts[idx][0].strip()]['line'], existing_performance[prompts[idx][0].strip()]['condition']))
            line_coverage = existing_performance[prompts[idx][0].strip()]['line']
            condition_coverage = existing_performance[prompts[idx][0].strip()]['condition']
        else:
            line_coverage, condition_coverage, _ = evaluate_coverage(cfg, prompts[idx][0], prompts[idx][1], str(100+len(existing_rules)))
            print('average line coverage: {}, average condition coverage: {}'.format(line_coverage, condition_coverage))
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(100+len(existing_rules)) + '/' +'performance.jsonl', 'a') as f:
                f.write_all([{'prompt':prompts[idx][0], 'idx':prompts[idx][1], 'line':line_coverage, 'condition':condition_coverage}])
        performance.append(line_coverage+condition_coverage)
   
    for idx in range(0, int(len(performance)/number)):
        if max(performance[idx*number:(idx+1)*number])>best_prompt_performance:
            effective_rules.append(rules[idx*number+performance[idx*number:(idx+1)*number].index(max(performance[idx*number:(idx+1)*number]))])
    return effective_rules



def fix_json_quotes(json_str):
    trimmed_string = json_str.strip()[1:-1]
    
    result_dict = {}
    key = None
    value = None
    in_quotes = False
    quote_count = 0
    current = []
    is_key = True
    
    for char in trimmed_string:
        if char == '"':
            in_quotes = not in_quotes
            quote_count += 1
            current.append(char)
        elif char == ',' and quote_count % 2 == 0:
            if key is not None and value is None:
                value = ''.join(current).strip()
                result_dict[key] = value
                key = None
                value = None
                current = []
                is_key = True
            continue
        elif char == ':' and quote_count % 2 == 0:
            key = ''.join(current).strip()
            current = []
            is_key = False
        else:
            current.append(char)
    
    if key is not None and value is None:
        value = ''.join(current).strip()
        result_dict[key] = value

    fixed_json_str = {}
    for k, v in result_dict.items():
        fixed_json_str[k.replace('"','')] = v.replace('"','')

    return fixed_json_str



def prompt_improvement_ape(cfg, selected_prompts, iteration_number, files, inducted_rules, ut_data, best_prompt_performance, selected_prompts_):
    print('improving prompt')
    success = 0
    fail_count = 0
    answers = []
    messages = " Generate a variation of the following instructions while keeping the semantic meaning\nInstruction 1:{}\n Instruction 2:{}\n Instruction 3:{}. Only reply with a instruction and do not answer others.".format(selected_prompts[0][0], selected_prompts[1][0], selected_prompts[2][0])
    if 'gpt' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        model = cfg.model_name
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, temperature=1)
                for answer in response["choices"]:
                    if '```' not in answer['message']['content']:
                        answers.append((answer['message']['content'].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    elif '```java' not in answer['message']['content']:
                        answers.append((answer['message']['content'].split('```')[1].split('```')[0].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    else:
                        answers.append(('```'.join(answer['message']['content'].split('```')[1:-1]).replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    print(answers[-1][0].strip())
                success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        model = cfg.model_name
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, temperature=1)
                for answer in response["choices"]:
                    if '```' not in answer['message']['content']:
                        answers.append((answer['message']['content'].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    elif '```java' not in answer['message']['content']:
                        answers.append((answer['message']['content'].split('```')[1].split('```')[0].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    else:
                        answers.append(('```'.join(answer['message']['content'].split('```')[1:-1]).replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    print(answers[-1][0].strip())
                success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    return answers, []

def prompt_improvement_opro(cfg, selected_prompts_, iteration_number, files, inducted_rules, ut_data, best_prompt_performance, selected_prompts):
    print('improving prompt')
    success = 0
    fail_count = 0
    answers = []
    messages = (
        "Your task is to generate the instruction <INS> for unit test generation."
        " Below are some previous instructions with their with their scores\n"
    )
    previous_prompts = '\n'
    for selected_prompt in selected_prompts:
        previous_prompts += f"instruction:{selected_prompt['prompt'][0]}  score:{selected_prompt['score']}\n"
    messages += previous_prompts
    messages += "Below are some problems.\n"
    messages += input_output_pair
    messages += (
        "\n\nGenerate a instruction that"
        " is different from all the instructions <INS> above,"
        " and has a higher score than all the instructions <INS> above."
        " The instruction should be concise, effective,"
        " and generally applicable to all problems above."
        " Please return with one instruction in block (``` ```)."
    )
    if 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        model = cfg.model_name
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, temperature=1)
                for answer in response["choices"]:
                    if '```' not in answer['message']['content']:
                        answers.append((answer['message']['content'].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    elif '```java' not in answer['message']['content']:
                        answers.append((answer['message']['content'].split('```')[1].split('```')[0].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    else:
                        answers.append(('```'.join(answer['message']['content'].split('```')[1:-1]).replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    print(answers[-1][0].strip())
                success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    elif 'gpt' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        model = cfg.model_name
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model=model, messages=messages, temperature=1)
                for answer in response.choices:
                    if '```' not in answer.message.content:
                        answers.append((answer.message.content.replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    elif '```java' not in answer.message.content:
                        answers.append((answer.message.content.split('```')[1].split('```')[0].replace('java','').strip(), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    else:
                        answers.append(('```'.join(answer.message.content.split('```')[1:-1].replace('java','').strip()), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    print(answers[-1][0].strip())
                success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    print(answers)
    return answers, []


def get_final_prompt(text):
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith("\"") and text.endswith("\""):
            text = text[1:-1]
        return text

def prompt_improvement_de(cfg, selected_prompts, iteration_number, files, inducted_rules, ut_data, best_prompt_performance, selected_prompts_):
    print('improving prompt')
    
    template = '''
    Please follow the instruction step-by-step to generate a better prompt.
    1. Identify the different parts between the Prompt 1 and Prompt 2:
    Prompt 1: <prompt1>
    Prompt 2: <prompt2>
    2. Randomly mutate the different parts
    3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
    Prompt 3: <prompt0>
    '''
    messages = (
        template.replace("<prompt0>", selected_prompts[0][0])
        .replace("<prompt1>", selected_prompts[1][0])
        .replace("<prompt2>", selected_prompts[2][0])
    )
    answers = []
    if 'gpt' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        model = cfg.model_name
        
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=1:
            try:
                n = cfg.generated_number
                for i in range(0, n):
                    response = openai.ChatCompletion.create(model= cfg.model_name, messages=messages, n = 1, temperature=1)
                    for answer in response["choices"]:
                        answers.append((get_final_prompt(answer.message.content.strip()), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                success=+1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
        print("new_prompt:")
        print(answers)
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        model = cfg.model_name
        
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model= cfg.model_name, messages=messages, temperature=1)
                for answer in response["choices"]:
                    if get_final_prompt(answer.message.content.strip()).strip() == '` and `':
                        continue
                    answers.append((get_final_prompt(answer.message.content.strip()), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
        print("new_prompt:")
        print(answers)

    return answers, []


def prompt_improvement_ga(cfg, selected_prompts, iteration_number, files, inducted_rules, ut_data, best_prompt_performance, selected_prompts_):
    print('improving prompt')
    
    answers = []
    template = '''
    Please follow the instruction step-by-step to generate a better prompt.
    1. Crossover the following prompts and generate a new prompt:
    Prompt 1: <prompt1>
    Prompt 2: <prompt2>
    2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.
    '''
    messages = (
        template
        .replace("<prompt1>", selected_prompts[0][0])
        .replace("<prompt2>", selected_prompts[1][0])
    )
    if 'gpt' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        model = cfg.model_name
        
        success = 0
        fail_count = 0
        while success!=1:
            messages = [
                    {"role": "user", "content": messages}
                    ]
            try:
                n = cfg.generated_number
                for i in range(0, n):
                    response = openai.ChatCompletion.create(model= cfg.model_name, messages=messages, n = 1, temperature=1)
                    for answer in response["choices"]:
                        answers.append((get_final_prompt(answer.message.content.strip()), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                success=+1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
        print("new_prompt:")
        print(answers)
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        model = cfg.model_name
        
        success = 0
        fail_count = 0
        messages = [
                {"role": "user", "content": messages}
                ]
        while success!=cfg.generated_number:
            try:
                response = openai.ChatCompletion.create(model= cfg.model_name, messages=messages, temperature=1)
                for answer in response["choices"]:
                    print(answer.message.content)
                    if get_final_prompt(answer.message.content.strip()).strip() == '` and `':
                        continue
                    answers.append((get_final_prompt(answer.message.content.strip()), cfg.seed_number+iteration_number*cfg.generated_number+len(answers)))
                    success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>100:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
        print("new_prompt:")
        print(answers)

    return answers, []


def generate_seed_prompt(cfg, prompts):
    print('generating seed prompt')
    success = 0
    fail_count = 0
    if 'gpt' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        while success!=1:
            messages = [
                {"role": "system", "content": "You are a prompt engineering and now you will help to write prompt. Make the prompt you write in block (``` ```). Do not explain anything and include any extra instructions, only print the prompt."},
                {"role": "user", "content": 'Below are input-output pairs that correspond to some underlying task:'+ input_output_pair + 'Please write the instruction that describes the task.'}
                ]
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, n=cfg.seed_number-len(prompts), temperature=1)
                answers = []
                for answer in response["choices"]:
                    if '```' not in answer['message']['content']:
                        answers.append((answer['message']['content'].strip()), len(prompts)+len(answers))
                    elif '```java' not in answer['message']['content']:
                        answers.append((answer['message']['content'].strip().split('```')[1].split('```')[0], len(prompts)+len(answers)))
                    else:
                        answers.append(('```'.join(answer['message']['content'].strip().split('```')[1:-1]), len(prompts)+len(answers)))
                    print(answers[-1][0].strip())
                success=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>10:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        openai.api_key = llm_api_key[0]
        openai.api_base = "https://api.deepinfra.com/v1/openai/chat/completions"
        while success!=cfg.seed_number-len(prompts):
            messages = [
                {"role": "system", "content": "You are a prompt engineering and now you will help to write prompt. Make the prompt you write in block (``` ```). Do not explain anything and include any extra instructions, only print the prompt."},
                {"role": "user", "content": 'Below are input-output pairs that correspond to some underlying task:'+ input_output_pair + 'Please write the instruction that describes the task.'}
                ]
            try:
                response = openai.ChatCompletion.create(model=cfg.model_name, messages=messages, temperature=1)
                answers = []
                for answer in response.choices:
                    if '```' not in answer.message.content:
                        answers.append((answer.message.content.strip()), len(prompts)+len(answers))
                    elif '```java' not in answer.message.content:
                        answers.append((answer.message.content.strip().split('```')[1].split('```')[0], len(prompts)+len(answers)))
                    else:
                        answers.append(('```'.join(answer.message.content.strip().split('```')[1:-1]), len(prompts)+len(answers)))
                    print(answers[-1][0].strip())
                success+=1
            except Exception as e:
                info = e.args[0]
                fail_count+=1
                if 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                print(info)
            if fail_count>10:
                print('{} fail more than 10 times'.format(str(fail_count)))
                break
    return prompts+answers


def main(cfg, prompts, ut_data, test_ut_data):
    if len(prompts)<cfg.seed_number:
       prompts = generate_seed_prompt(cfg, prompts)
    
    cfg.stage = 'train'
    inducted_rules = []
    for ite in range(cfg.iteration_number):
        if len(inducted_rules)>0:
            for i in range(len(prompts)):
                prompts[i] = (prompts[i][0].strip()+' Specifically, please follow the following rules when writing test cases: ' + ' '.join([f'{idx+1}. {rule[0]}' for idx, rule in enumerate(inducted_rules)]),prompts[i][1])
        
        print('############################# ', ite)
        results = []
        lines = []
        conditions = []
        ut_generation(cfg, ut_data, prompts, ite)
        existing_performance = {}
        if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'performance.jsonl'):
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'performance.jsonl') as f:
                for item in f:
                    existing_performance[item['prompt'].strip()] = item
        for idx in range(len(prompts)):
            print('prompt: {}'.format(prompts[idx][0].strip()))
            if prompts[idx][0].strip() in existing_performance:
                print('average line coverage: {}, average condition coverage: {}'.format(existing_performance[prompts[idx][0].strip()]['line'], existing_performance[prompts[idx][0].strip()]['condition']))
                line_coverage = existing_performance[prompts[idx][0].strip()]['line']
                condition_coverage = existing_performance[prompts[idx][0].strip()]['condition']
            else:
                line_coverage, condition_coverage, _ = evaluate_coverage(cfg, prompts[idx][0], prompts[idx][1], str(ite))
                print('average line coverage: {}, average condition coverage: {}'.format(line_coverage, condition_coverage))
                with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'performance.jsonl', 'a') as f:
                    f.write_all([{'prompt':prompts[idx][0], 'idx':prompts[idx][1], 'line':line_coverage, 'condition':condition_coverage}])
            lines.append(line_coverage)
            conditions.append(condition_coverage)
            results.append(line_coverage+condition_coverage)
            prompts[idx] = prompts[idx] + (line_coverage+condition_coverage,)
        print('best line coverage: {}, best condition coverage: {}'.format(max(lines), max(conditions)))
        
        indices = sorted(range(len(results)), key=lambda i: results[i], reverse=True)[:cfg.seed_number-cfg.generated_number]

        selected_prompts_improve = []
        for i in indices:
            selected_prompt = {}
            selected_prompt['prompt'] = prompts[i]
            selected_prompt['score'] = results[i]
            selected_prompts_improve.append(selected_prompt)

        selected_prompts = []
        for i in indices:
            selected_prompts.append((prompts[i][0].split('Specifically, please follow the following rules when writing test cases')[0].strip(), prompts[i][1]))
        selected_files = [os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite), 'processed_prediction'+str(selected_prompts[0][1])+'.jsonl')]
        if ite != cfg.iteration_number-1:
            selected_files = []
            for prompt in selected_prompts:
                i=prompt[1]
                selected_files.append(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite), 'processed_prediction'+str(i)+'.jsonl'))
                source_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite), 'processed_prediction'+str(i)+'.jsonl')
                destination_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1), 'processed_prediction'+str(i)+'.jsonl')
                if not os.path.exists(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1))):
                    os.makedirs(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1)))
                shutil.copy2(source_file, destination_file)
        source_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite), 'prediction.jsonl')
        destination_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1), 'prediction.jsonl')
        if not os.path.exists(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1))):
           os.makedirs(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1)))
        if not os.path.exists(destination_file):
            shutil.copy2(source_file, destination_file)
        source_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite), 'performance.jsonl')
        destination_file = os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1), 'performance.jsonl')
        if not os.path.exists(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1))):
           os.makedirs(os.path.join(cfg.output_base_dir, cfg.mode, cfg.stage, str(ite+1)))
        if not os.path.exists(destination_file):
            shutil.copy2(source_file, destination_file)

        if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'new_prompt.jsonl'):
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'new_prompt.jsonl') as f:
                for obj in f:
                    print("prompt", obj['prompt'], obj['idx'])
                    selected_prompts.append((obj['prompt'], obj['idx']))
            prompts = selected_prompts
        else:
            prompts = selected_prompts
            if 'evol_de' in cfg.mode:
                new_prompts, _ = prompt_improvement_de(cfg, selected_prompts, ite, selected_files, inducted_rules, ut_data, max(results), selected_prompts_improve)
            elif 'evol_ga' in cfg.mode:
                new_prompts, _ = prompt_improvement_ga(cfg, selected_prompts, ite, selected_files, inducted_rules, ut_data, max(results), selected_prompts_improve)
            elif 'ape' in cfg.mode:
                new_prompts, _ = prompt_improvement_ape(cfg, selected_prompts, ite, selected_files, inducted_rules, ut_data, max(results), selected_prompts_improve)
            elif 'opro' in cfg.mode:
                new_prompts, _ = prompt_improvement_opro(cfg, selected_prompts, ite, selected_files, inducted_rules, ut_data, max(results), selected_prompts_improve)
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'new_prompt.jsonl', 'w') as f:
                write_data = []
                for prompt in new_prompts:
                    write_data.append({'prompt':prompt[0], 'idx':prompt[1]})
                f.write_all(write_data)
            prompts = selected_prompts + new_prompts
                
    
        if os.path.exists(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'inducted_rules.jsonl'):
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'inducted_rules.jsonl') as f:
                for obj in f:
                    print("inducted_rules", obj['rule'], obj['center'])
                    if (obj['rule'], obj['center']) not in inducted_rules:
                        inducted_rules.append((obj['rule'], obj['center']))
        else:
            inducted_rules = []
            print("New rules:")
            print(inducted_rules)
            with jsonlines.open(cfg.output_base_dir+ '/' + cfg.mode + '/' + cfg.stage + '/' + str(ite) + '/' +'inducted_rules.jsonl', 'w') as f:
                write_data = []
                for rule in inducted_rules:
                    write_data.append({'rule':rule[0], 'center':rule[1]})
                f.write_all(write_data)


    print('testing stage')
    cfg.stage = 'test'
    lines = []
    conditions = []
    ut_generation(cfg, test_ut_data, prompts, 0)
    print('prompt: {}'.format(prompts[0]))
    line_coverage, condition_coverage, _ = evaluate_coverage(cfg, prompts[0][0], prompts[0][1], '0')
    print('best line coverage: {}, best condition coverage: {}'.format(line_coverage, condition_coverage))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--test_data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_base_dir", default=None, type=str, required=True)
    parser.add_argument("--max_test_cases", default=10, type=int, required=True)
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--seed_number", default=5, type=int, required=True)
    parser.add_argument("--generated_number", default=2, type=int, required=True)
    parser.add_argument("--iteration_number", default=5, type=int, required=True)
    parser.add_argument("--seed_prompt_addr", default='seed_prompt.txt', type=str, required=True)

    cfg = parser.parse_args()
    if 'gpt' in cfg.model_name:
        llm_api_key = gpt_api_keys
    elif 'llama' in cfg.model_name or 'Qwen' in cfg.model_name:
        llm_api_key = deepinfra_api_keys
        
    with open(cfg.seed_prompt_addr) as f:
        prompts = f.readlines()
    seed_prompt = []
    for prompt in prompts[:cfg.seed_number]:
        if len(prompt.strip())>0:
            seed_prompt.append((prompt, len(seed_prompt)))
    ut_data = []
    with jsonlines.open(cfg.data_dir) as f:
        for obj in f:
            ut_data.append(obj)

    test_ut_data = []
    with jsonlines.open(cfg.test_data_dir) as f:
        for obj in f:
            test_ut_data.append(obj)

    main(cfg, seed_prompt, ut_data, test_ut_data)


