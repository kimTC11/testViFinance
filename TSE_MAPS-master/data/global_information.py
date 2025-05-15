import os
import re
import jsonlines
from tqdm import tqdm
from tree_sitter import Language, Parser
JAVA_LANGUAGE = Language('./my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

base_dir = '/data/ubuntu/ut_gen/d4j_projects'
file_path={'Cli':{
               1: "org.apache.commons.cli.CommandLine",
               2: "org.apache.commons.cli.PosixParser",
               3: "org.apache.commons.cli.TypeHandler",
               4: "org.apache.commons.cli.Parser",
               5: "org.apache.commons.cli.Util",
               7: "org.apache.commons.cli2.builder.PatternBuilder",
               8: "org.apache.commons.cli.HelpFormatter",
               9: "org.apache.commons.cli.Parser",
               10: "org.apache.commons.cli.Parser",
               11: "org.apache.commons.cli.HelpFormatter",
               12: "org.apache.commons.cli.GnuParser",
               13: "org.apache.commons.cli2.WriteableCommandLine;org.apache.commons.cli2.commandline.WriteableCommandLineImpl;org.apache.commons.cli2.option.ArgumentImpl",
               14: "org.apache.commons.cli2.option.GroupImpl",
               15: "org.apache.commons.cli2.commandline.WriteableCommandLineImpl",
               16: "org.apache.commons.cli2.Option;org.apache.commons.cli2.commandline.WriteableCommandLineImpl;org.apache.commons.cli2.option.GroupImpl;org.apache.commons.cli2.option.OptionImpl",
               17: "org.apache.commons.cli.PosixParser",
               18: "org.apache.commons.cli.PosixParser",
               19: "org.apache.commons.cli.PosixParser",
               20: "org.apache.commons.cli.PosixParser",
               21: "org.apache.commons.cli2.WriteableCommandLine;org.apache.commons.cli2.commandline.WriteableCommandLineImpl;org.apache.commons.cli2.option.GroupImpl",
               22: "org.apache.commons.cli.PosixParser",
               23: "org.apache.commons.cli.HelpFormatter",
               24: "org.apache.commons.cli.HelpFormatter",
               25: "org.apache.commons.cli.HelpFormatter",
               26: "org.apache.commons.cli.OptionBuilder",
               27: "org.apache.commons.cli.OptionGroup",
               28: "org.apache.commons.cli.Parser",
               29: "org.apache.commons.cli.Util",
               30: "org.apache.commons.cli.DefaultParser;org.apache.commons.cli.Parser",
               31: "org.apache.commons.cli.HelpFormatter;org.apache.commons.cli.Option;org.apache.commons.cli.OptionBuilder",
               32: "org.apache.commons.cli.HelpFormatter",
               33: "org.apache.commons.cli.HelpFormatter",
               34: "org.apache.commons.cli.Option;org.apache.commons.cli.OptionBuilder",
               35: "org.apache.commons.cli.Options",
               36: "org.apache.commons.cli.OptionGroup;org.apache.commons.cli.Options",
               37: "org.apache.commons.cli.DefaultParser",
               38: "org.apache.commons.cli.DefaultParser",
               39: "org.apache.commons.cli.TypeHandler",
               40: "org.apache.commons.cli.TypeHandler"
           },
           'Chart':{
                1: "org.jfree.chart.renderer.category.AbstractCategoryItemRenderer",
                2: "org.jfree.data.general.DatasetUtilities",
                3: "org.jfree.data.time.TimeSeries",
                4: "org.jfree.chart.plot.XYPlot",
                5: "org.jfree.data.xy.XYSeries",
                6: "org.jfree.chart.util.ShapeList",
                7: "org.jfree.data.time.TimePeriodValues",
                8: "org.jfree.data.time.Week",
                9: "org.jfree.data.time.TimeSeries",
                10: "org.jfree.chart.imagemap.StandardToolTipTagFragmentGenerator",
                11: "org.jfree.chart.util.ShapeUtilities",
                12: "org.jfree.chart.plot.MultiplePiePlot",
                13: "org.jfree.chart.block.BorderArrangement",
                14: "org.jfree.chart.plot.CategoryPlot;org.jfree.chart.plot.XYPlot",
                15: "org.jfree.chart.plot.PiePlot",
                16: "org.jfree.data.category.DefaultIntervalCategoryDataset",
                17: "org.jfree.data.time.TimeSeries",
                18: "org.jfree.data.DefaultKeyedValues;org.jfree.data.DefaultKeyedValues2D",
                19: "org.jfree.chart.plot.CategoryPlot",
                20: "org.jfree.chart.plot.ValueMarker",
                21: "org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset",
                22: "org.jfree.data.KeyedObjects2D",
                23: "org.jfree.chart.renderer.category.MinMaxCategoryRenderer",
                24: "org.jfree.chart.renderer.GrayPaintScale",
                25: "org.jfree.chart.renderer.category.StatisticalBarRenderer",
                26: "org.jfree.chart.axis.Axis"
            },
           'Csv':{
                1: "org.apache.commons.csv.ExtendedBufferedReader",
                2: "org.apache.commons.csv.CSVRecord",
                3: "org.apache.commons.csv.Lexer",
                4: "org.apache.commons.csv.CSVParser",
                5: "org.apache.commons.csv.CSVPrinter",
                6: "org.apache.commons.csv.CSVRecord",
                7: "org.apache.commons.csv.CSVParser",
                8: "org.apache.commons.csv.CSVFormat",
                9: "org.apache.commons.csv.CSVRecord",
                10: "org.apache.commons.csv.CSVPrinter",
                11: "org.apache.commons.csv.CSVParser",
                12: "org.apache.commons.csv.CSVFormat",
                13: "org.apache.commons.csv.CSVFormat;org.apache.commons.csv.CSVPrinter",
                14: "org.apache.commons.csv.CSVFormat",
                15: "org.apache.commons.csv.CSVFormat",
                16: "org.apache.commons.csv.CSVParser"
            },
           'Gson':{
                1: "com.google.gson.TypeInfoFactory",
                2: "com.google.gson.internal.bind.TypeAdapters",
                3: "com.google.gson.internal.ConstructorConstructor",
                4: "com.google.gson.stream.JsonReader;com.google.gson.stream.JsonWriter",
                5: "com.google.gson.internal.bind.util.ISO8601Utils",
                6: "com.google.gson.internal.bind.JsonAdapterAnnotationTypeAdapterFactory",
                7: "com.google.gson.stream.JsonReader",
                8: "com.google.gson.internal.UnsafeAllocator",
                9: "com.google.gson.internal.bind.JsonTreeWriter;com.google.gson.internal.bind.TypeAdapters;com.google.gson.stream.JsonWriter",
                10: "com.google.gson.internal.bind.ReflectiveTypeAdapterFactory",
                11: "com.google.gson.internal.bind.TypeAdapters",
                12: "com.google.gson.internal.bind.JsonTreeReader",
                13: "com.google.gson.stream.JsonReader",
                14: "com.google.gson.internal.$Gson$Types",
                15: "com.google.gson.stream.JsonWriter",
                16: "com.google.gson.internal.$Gson$Types",
                17: "com.google.gson.DefaultDateTypeAdapter",
                18: "com.google.gson.internal.$Gson$Types"
            },
           'Lang':{
                1: "org.apache.commons.lang3.math.NumberUtils",
                3: "org.apache.commons.lang3.math.NumberUtils",
                4: "org.apache.commons.lang3.text.translate.LookupTranslator",
                5: "org.apache.commons.lang3.LocaleUtils",
                6: "org.apache.commons.lang3.text.translate.CharSequenceTranslator",
                7: "org.apache.commons.lang3.math.NumberUtils",
                8: "org.apache.commons.lang3.time.FastDatePrinter",
                9: "org.apache.commons.lang3.time.FastDateParser",
                10: "org.apache.commons.lang3.time.FastDateParser",
                11: "org.apache.commons.lang3.RandomStringUtils",
                12: "org.apache.commons.lang3.RandomStringUtils",
                13: "org.apache.commons.lang3.SerializationUtils",
                14: "org.apache.commons.lang3.StringUtils",
                15: "org.apache.commons.lang3.reflect.TypeUtils",
                16: "org.apache.commons.lang3.math.NumberUtils",
                17: "org.apache.commons.lang3.text.translate.CharSequenceTranslator",
                18: "org.apache.commons.lang3.time.FastDateFormat",
                19: "org.apache.commons.lang3.text.translate.NumericEntityUnescaper",
                20: "org.apache.commons.lang3.StringUtils",
                21: "org.apache.commons.lang3.time.DateUtils",
                22: "org.apache.commons.lang3.math.Fraction",
                23: "org.apache.commons.lang3.text.ExtendedMessageFormat",
                24: "org.apache.commons.lang3.math.NumberUtils",
                25: "org.apache.commons.lang3.text.translate.EntityArrays",
                26: "org.apache.commons.lang3.time.FastDateFormat",
                27: "org.apache.commons.lang3.math.NumberUtils",
                28: "org.apache.commons.lang3.text.translate.NumericEntityUnescaper",
                29: "org.apache.commons.lang3.SystemUtils",
                30: "org.apache.commons.lang3.StringUtils",
                31: "org.apache.commons.lang3.StringUtils",
                32: "org.apache.commons.lang3.builder.HashCodeBuilder",
                33: "org.apache.commons.lang3.ClassUtils",
                34: "org.apache.commons.lang3.builder.ToStringStyle",
                35: "org.apache.commons.lang3.ArrayUtils",
                36: "org.apache.commons.lang3.math.NumberUtils",
                37: "org.apache.commons.lang3.ArrayUtils",
                38: "org.apache.commons.lang3.time.FastDateFormat",
                39: "org.apache.commons.lang3.StringUtils",
                40: "org.apache.commons.lang.StringUtils",
                41: "org.apache.commons.lang.ClassUtils",
                42: "org.apache.commons.lang.Entities",
                43: "org.apache.commons.lang.text.ExtendedMessageFormat",
                44: "org.apache.commons.lang.NumberUtils",
                45: "org.apache.commons.lang.WordUtils",
                46: "org.apache.commons.lang.StringEscapeUtils",
                47: "org.apache.commons.lang.text.StrBuilder",
                48: "org.apache.commons.lang.builder.EqualsBuilder",
                49: "org.apache.commons.lang.math.Fraction",
                50: "org.apache.commons.lang.time.FastDateFormat",
                51: "org.apache.commons.lang.BooleanUtils",
                52: "org.apache.commons.lang.StringEscapeUtils",
                53: "org.apache.commons.lang.time.DateUtils",
                54: "org.apache.commons.lang.LocaleUtils",
                55: "org.apache.commons.lang.time.StopWatch",
                56: "org.apache.commons.lang.time.FastDateFormat",
                57: "org.apache.commons.lang.LocaleUtils",
                58: "org.apache.commons.lang.math.NumberUtils",
                59: "org.apache.commons.lang.text.StrBuilder",
                60: "org.apache.commons.lang.text.StrBuilder",
                61: "org.apache.commons.lang.text.StrBuilder",
                62: "org.apache.commons.lang.Entities",
                63: "org.apache.commons.lang.time.DurationFormatUtils",
                64: "org.apache.commons.lang.enums.ValuedEnum",
                65: "org.apache.commons.lang.time.DateUtils"
            }
           }



def find_extend_class_line(text, class_name):
    pattern = r'(public|private)\s+.*?(extends\s+' + re.escape(class_name) + r'.*?{)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()


def remove_comments(java_code):
    single_line_comment_pattern = r'//.*?$'
    multi_line_comment_pattern = r'/\*.*?\*/'
    doc_comment_pattern = r'/\*\*.*?\*/'
    combined_pattern = f'({single_line_comment_pattern})|({multi_line_comment_pattern})|({doc_comment_pattern})'
    code_without_comments = re.sub(combined_pattern, '', java_code, flags=re.DOTALL | re.MULTILINE)

    return code_without_comments

def find_subclass(project, id, method, skip_full_name, source):
    if project=='Gson' and id in [14,16,18]:
        new_method = 'public final class $Gson$'+method.strip()
        return '', new_method
    
    file_names = os.path.join(base_dir, f'{project}_{id}_buggy', source, file_path[project][id].replace('.', '/')+'.java')
    not_found=True
    for file_name in file_names.split(';'):
        if file_name.split('/')[-1].replace('.java', '').strip() != class_name:
            continue
        not_found = False
        if '.java' not in file_name:
            file_name = os.path.join(base_dir, f'{project}_{id}_buggy', source, file_name+'.java')
        else:
            file_name = os.path.join(base_dir, f'{project}_{id}_buggy', source, file_name)
        with open(file_name) as f:
            content = f.read()
        content = remove_comments(content)

        # find full name
        if not skip_full_name:
            method_sig = method.split('{')[0].strip()+' {'
            method_name = class_name
            
            pattern = re.sub(r'\s+', r'\\s*', method_sig)
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                print(file_name)
                print(content)
                print('################################')
                print(method_sig)
                input()
            else:
                start, end = match.span()
                matched_part = content[start:end]
                before_lines = content[:start].splitlines()
                after_lines = content[end:].splitlines()
                start_line_index = len(before_lines)
                end_line_index = len(before_lines) + len(content[start:end].splitlines()) - 1
                all_lines = content.splitlines()
                full_name =  '\n'.join(all_lines[start_line_index-1:end_line_index])
            
                new_method = method.replace(method_sig, full_name)
        else:
            full_name = method.split('{')[0].strip()+' {'
            new_method = method

        if 'abstract' in full_name or 'private' in full_name:
            superclasses = []
            for dirpath, dirnames, filenames in os.walk(os.path.join(base_dir, f'{project}_{id}_buggy', source)):
                for filename in filenames:
                    item_path = os.path.join(dirpath, filename)
                    if item_path.endswith('.java'):
                        try:
                            with open(os.path.join(os.path.dirname(file_name), item_path)) as f:
                                content = f.read()
                        except:
                            continue
                        content = remove_comments(content)
                        superclass = find_extend_class_line(content, method_name)
                        if superclass is not None:
                            superclasses.append('// '+superclass.replace('\n', ' '))
            if len(superclasses) == 0:
                print(matched_part)
                print(full_name)
                print(file_name)
                return '', new_method
            else:
                return '// Avaible SubClasses:\n'+'\n'.join(superclasses)+'\n', new_method
        else:
            return '', new_method
    if not_found:
        print(project, id, method)
        print('#'*40)
        return '', method


def find_variables(node, code):
    if node.type == 'formal_parameters':
        return get_variables(node,code)
    for child in node.children:
        self_defined = find_variables(child,code)
        if self_defined is not None:
            return self_defined


def get_variables(node, code):
    self_defined = []
    for child in node.children:
        if child.type=='formal_parameter' and child.children[0].type=='type_identifier':
            self_defined.append({'type':code[child.children[0].start_point[1]:child.children[0].end_point[1]], 'name':code[child.children[1].start_point[1]:child.children[1].end_point[1]]})
    return self_defined


def find_construction(node, code, class_name, constructor):
    if node.type=='class_declaration':
        constructor.append(code[node.start_point[1]:node.end_point[1]].split('{')[0].replace('\n', ' ')+'{')
    if node.type == 'constructor_declaration' and code[node.children[1].start_point[1]:node.children[1].end_point[1]].strip()==class_name:
        constructor.append(code[node.start_point[1]:node.end_point[1]].replace('\n', ' '))
    for child in node.children:
        find_construction(child, code, class_name, constructor)

data= []
with jsonlines.open('all_bsl.jsonl') as f:
    for obj in f:
        data.append(obj)
print(len(data))
processed_data = []
for obj in tqdm(data):
    project = obj["project"]
    id = obj['id']
    method = obj['focal_method_with_context']
    class_name = method.split()[0].strip()
    if project == 'Gson':
        source='gson/src/main/java'
    elif project == 'Chart':
        source='source'
    elif project == 'Cli' and id>=30:
        source='src/main/java'
    elif project == 'Cli':
        source='src/java'
    elif project == 'Lang' and id>=36:
        source='src/java'
    else:
        source='src/main/java'
    global_info, new_method = find_subclass(project, id, method, False, source)
    global_info = re.sub(r' +', ' ', global_info)

    tree = parser.parse(bytes(new_method.replace('\n', ' '),'utf8'))
    root_node = tree.root_node
    self_defined = find_variables(root_node, new_method.replace('\n', ' '))
    construction = ''
    for variable in self_defined:
        for dirpath, dirnames, filenames in os.walk(os.path.join(base_dir, f'{project}_{id}_buggy', source)):
            for filename in filenames:
                item_path = os.path.join(dirpath, filename)
                if item_path.endswith('/'+variable['type']+'.java'):
                    with open(os.path.join(base_dir, f'{project}_{id}_buggy', source, item_path)) as f:
                        content = f.read()
                    content = remove_comments(content)
                    tree = parser.parse(bytes(content.replace('\n', ' '),'utf8'))
                    root_node = tree.root_node
                    constructor = []
                    find_construction(root_node,content.replace('\n', ' '), variable['type'], constructor)
                    if len(constructor)>0:
                       construction += '// '+'\n'.join(constructor).strip()+'}\n'
    if len(construction)>0:
        construction = '// Class definition of input parameters:\n'+construction
        construction = re.sub(r' +', ' ', construction)
    obj['method_global_context'] = '// Focal method:\n'+new_method+global_info+construction
    processed_data.append(obj)

                    


    

print(len(processed_data))
with jsonlines.open('all_ours.jsonl','w') as f:
    f.write_all(processed_data)





data= []
with jsonlines.open('sample_bsl.jsonl') as f:
    for obj in f:
        data.append(obj)
print(len(data))
processed_data = []
for obj in tqdm(data):
    project = obj["project"]
    id = obj['id']
    method = obj['focal_method_with_context']
    class_name = method.split()[0].strip()
    if project == 'Gson':
        source='gson/src/main/java'
    elif project == 'Chart':
        source='source'
    elif project == 'Cli' and id>=30:
        source='src/main/java'
    elif project == 'Cli':
        source='src/java'
    elif project == 'Lang' and id>=36:
        source='src/java'
    else:
        source='src/main/java'
    global_info, new_method = find_subclass(project, id, method, False, source)
    global_info = re.sub(r' +', ' ', global_info)

    tree = parser.parse(bytes(new_method.replace('\n', ' '),'utf8'))
    root_node = tree.root_node
    self_defined = find_variables(root_node, new_method.replace('\n', ' '))
    construction = ''
    for variable in self_defined:
        for dirpath, dirnames, filenames in os.walk(os.path.join(base_dir, f'{project}_{id}_buggy', source)):
            for filename in filenames:
                item_path = os.path.join(dirpath, filename)
                if item_path.endswith('/'+variable['type']+'.java'):
                    with open(os.path.join(base_dir, f'{project}_{id}_buggy', source, item_path)) as f:
                        content = f.read()
                    content = remove_comments(content)
                    tree = parser.parse(bytes(content.replace('\n', ' '),'utf8'))
                    root_node = tree.root_node
                    constructor = []
                    find_construction(root_node,content.replace('\n', ' '), variable['type'], constructor)
                    if len(constructor)>0:
                       construction += '// '+'\n'.join(constructor).strip()+'}\n'
    if len(construction)>0:
        construction = '// Class definition of input parameters:\n'+construction
        construction = re.sub(r' +', ' ', construction)
    obj['method_global_context'] = '// Focal method:\n'+new_method+global_info+construction
    processed_data.append(obj)

                    


    

print(len(processed_data))
with jsonlines.open('sample_ours.jsonl','w') as f:
    f.write_all(processed_data)

