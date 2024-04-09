import re
import os
import tiktoken
import pandas as pd
import csv
import json
import nltk
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# os.environ["http_proxy"] = "http://127.0.0.1:10809"
# os.environ["https_proxy"] = "http://127.0.0.1:10809"

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
    # encoding = tiktoken.encoding_for_model('gpt-4')
    num_tokens = len(encoding.encode(text))
    return num_tokens


def get_input_pdfs(input_dir):
    input_pdfs = [file for file in os.listdir(input_dir) if file.endswith('.pdf')]
    return input_pdfs


def get_cache_files(cache_dir):
    cache_files = [file for file in os.listdir(cache_dir) if file.endswith('.json')]
    return cache_files


def save_failed_pdfs(failed_pdfs, failed_dir):
    """
    Save the paths of failed PDFs in a specified directory.
    """
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)

    with open(os.path.join(failed_dir, 'failed_jsons.txt'), 'w') as f:
        for pdf in failed_pdfs:
            f.write(f"{pdf}\n")


prompt = """
You will be provided with text containing detailed information on various materials used in chemical reactions.
Your task is to carefully read the text and extract specific information about each material.

Please follow the procedure below to find the following keywords in the text.

'Gibbs free energy', 'ΔGH*', 'ΔGOH*', 'ΔGOOH*', 'ΔGO*', 'overpotential', 'limiting potential', 'band gap', 'electron affinity', 'electronegativity', 'Theoretical exchange current', 'charge transfer number'
The data corresponding to the main search keywords.
Step 1. Search the text for a keyword.
Step 2. Search scope: Search the corresponding data near the keyword. The search scope is the sentence in which the keyword is located and the two before and after the sentence, a total of three sentences, ending with a period, excluding cross-line and commas.
Step 3. In this range, if there are multiple keywords that correspond to more than one piece of data, then all the data should be extracted and represented in multiple lines.
Step 4. No data: If there is no data, skip it and search for the next keyword location.
Step 5. If there is data, the material name and reaction type are searched in the data, if there is, it is extracted, and if there is not, it is skipped.
Step 6. Repeat processes 3, 4, and 5 until the full text search is complete.

The method of extraction remains the same, only note the sign of the data.

example：
The process with the largest ΔG was considered the ratedetermining step (RDS) for the OER. The third step (OOH* formation) was identified as the RDS of the AEM mechanism on the Co active site for Ov-N-Co3O4 and Ov-Co3O4 (Fig. 9a). In the conventional AEM, the Gibbs free energies of RDS for Ov-N-Co3O4 and Ov-Co3O4 were calculated to be 1.86 eV and 2.63 eV, respectively. These results indicated that N doping can regulate the electronic state of central Co atom and optimize the adsorption energies of intermediates, reducing the RDS of OER reaction.
Step 1.Search for the keyword Gibbs free energy in the text.
Step 2. Search for Gibbs free energy in three sentences. 
Step 3. Extract the data corresponding to the keyword, 1.86 eV, 2.63 eV.
Step 4. Extract the data corresponding to the material name and reaction type, Ov-N-Co3O4 corresponds to -1.86 eV, Ov-Co3O4 corresponds to 2.63 eV, and the reaction type is OER.
Step 5. The final results are
| Material Name | Reaction Type | Gibbs Free Energy | ΔGO* | ΔGOH* | ΔGOOH* | ΔGH* | Overpotential | Limiting Potential | Band Gap | Electron Affinity | Electronegativity | Theoretical Exchange Current | Charge Transfer Number |
| Ov-N-Co3O4 | OER | 1.86 eV | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ov-Co3O4 | OER | 2.63 eV | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

For each data, create a table with the following columns:
Material Name: compound name or chemical formula, e.g. V-CoOOH/CoS2, Ni@BN-yne, Ov-N-Co3O4.
Reaction Type: type of reaction, such as Oxygen Reduction Reaction(ORR), Oxygen Evolution Reaction(OER), Hydrogen Evolution Reaction(HER), Hydrogen Oxidation Reaction(HOR), etc..
Gibbs free energy, or ΔG, Gibbs free energies; the unit is eV.  
ΔGO*: or ΔG*O, gibbs free energy of adsorption O or *O.
ΔGOH*: or ΔG*OH, gibbs free energy of adsorption OH or *OH.
ΔGOOH*: or ΔGOO*, ΔG*OOH, gibbs free energy of adsorption OOH or *OOH.
ΔGH*: or ΔG*H, gibbs free energy of adsorption H or *H. e.g.The Gibbs free energy changes of H adsorption of HC1, HB1, and HNi are 0.14, 0.32, and 0.64 eV, respectively.
Overpotential: abbreviation \eta, the unit is V, eV or mV, it refers to the part of the electrocatalytic reaction that the actual voltage required to reach a certain current density exceeds the theoretical voltage. e.g. 0.97 eV.
Limiting Potential: abbreviation UL, the unit is V, eV or mV. The limit value of the potential difference in the battery reaction. e.g. 1.51 eV.
Electronegativity: A relative scale of the ability of atoms to attract electrons of each element in the periodic table. e.g. N (Nitrogen): 3.04.
Band gap: or bandgap, e.g. 2.90 eV.
Electron affinity: the ability of an atom or molecule to donate or receive electrons to the outside.
Theoretical exchange current: The electrode reaction is in equilibrium, and the current density corresponding to the cathode and anode can describe the ability of the electrode to gain and lose electrons in the reaction, and the symbol is io and o is the subscript.
Charge Transfer Number: The number of electrons gained and lost in a chemical reaction, is generally expressed by n.

If you find the relevant information, present it in a tabular format like this:
| Material Name | Reaction Type | Gibbs Free Energy | ΔGO* | ΔGOH* | ΔGOOH* | ΔGH* | Overpotential | Limiting Potential | Band Gap | Electron Affinity | Electronegativity | Theoretical Exchange Current | Charge Transfer Number |
| --------------- | ------- | ----------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |------- | ------- |

If you can not find any of the relevant data, just answer 'N/A' like this:
N/A

"""
prompt_tokens = count_tokens(prompt)
num_tokens_gpt_3_5 = 3000


# 纠正换行问题
def correct_line_breaks(text):
    text = re.sub(r' \n', ' ', text)
    text = re.sub(r'(\w)\-\n(\w)', r'\1\2', text)  # 连接被错误分割的词汇
    text = re.sub(r'\s+', ' ', text)
    return text


# 去除特殊字符和乱码
def remove_special_characters(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 去除非ASCII字符
    return text


def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r'(Notes and references|References|REFERENCES|Acknowledgment|ACKNOWLEDGMENT|Acknowledgements)'
    match = re.search(pattern, pdf_text)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.', '\[[\d\w]{1,3}\].+?[\d]{3,5};', '\([\d\w]{1,3}\).+?[\d]{3,5}\.',
            '\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},', '\[[\d\w]{1,3}\].+?[\d]{3,5}', '[\d\w]{1,3}\).+?[\d]{3,5}\.',
            '[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}', '^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(
                ',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(
                    re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(
                    matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text


def expend_keywords(keywords):
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    def get_synonyms(word):
        """获取单词的同义词"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def lemmatize_keywords(keywords):
        """将关键词列表词形还原"""
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in keywords]

    # 词形还原
    lemmatized_keywords = lemmatize_keywords(keywords)

    # 同义词扩展
    expanded_keywords = set(lemmatized_keywords)
    for keyword in lemmatized_keywords:
        expanded_keywords.update(get_synonyms(keyword))
    expanded_keywords = keywords + list(expanded_keywords)

    # 将关键词列表转换为小写
    keywords_lower = [keyword.lower() for keyword in expanded_keywords]

    expanded_keywords = list(set(keywords_lower))

    return expanded_keywords


def contain_keywords(text, keywords, need_expend=False):
    if need_expend:
        keywords = expend_keywords(keywords)
    keywords_lower = [keyword.lower() for keyword in keywords]

    if any(keyword in text.lower() for keyword in keywords_lower):
        return True

    return False


def filter_by_keywords(list_of_texts, keywords, need_expend=False):
    if need_expend:
        keywords = expend_keywords(keywords)
    keywords_lower = [keyword.lower() for keyword in keywords]

    filtered_texts = []
    for text in list_of_texts:
        if any(keyword in text.lower() for keyword in keywords_lower):
            filtered_texts.append(text)
    return filtered_texts


def split_strings(input_string, num_tokens, overlap=0):
    """Splits a string into chunks based on a maximum token count. """

    MAX_TOKENS = num_tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[:cut_index + 1].strip())
            current_string = current_string[cut_index + 1 - overlap:].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings


def merge_contents(input_data, num_tokens):
    return_data = []
    current_str = []
    currentd_pages = []
    current_tokens = 0
    for data in input_data:
        text = data['text']
        num_page = data['page_num']
        current_tokens += count_tokens(text)
        if current_tokens > num_tokens:
            tmp_str = '\n\n'.join(current_str)
            currentd_pages = list(set(currentd_pages))
            currentd_pages = map(str, currentd_pages)
            tmp_pages = ','.join(currentd_pages)
            tmp_tokens = count_tokens(tmp_str)
            return_data.append({
                'text': tmp_str,
                'page_num': tmp_pages,
                'num_tokens': tmp_tokens,
            })
            current_tokens = count_tokens(text)
            current_str = [text]
            currentd_pages = [num_page]
        else:
            current_str.append(text)
            currentd_pages.append(num_page)

    if current_tokens:
        tmp_str = '\n\n'.join(current_str)
        currentd_pages = list(set(currentd_pages))
        currentd_pages = map(str, currentd_pages)
        tmp_pages = ','.join(currentd_pages)
        tmp_tokens = count_tokens(tmp_str)
        return_data.append({
            'text': tmp_str,
            'page_num': tmp_pages,
            'num_tokens': tmp_tokens,
        })

    return return_data

def extract_doi(text):
    doi_pattern = r'10.\d{4,9}/[-._;()/:A-Z0-9]+'
    dois = re.findall(doi_pattern, text, flags=re.IGNORECASE)
    return dois
    # return dois[0]


def pre_process_for_3(json_data, key_words):
    file_name = json_data['file_name']
    data = json_data['data']
    return_data = []
    dois = []

    key_words = expend_keywords(key_words)
    # 3 的pdf文本提取方法是参考了pdf布局
    for d in data:
        if d['type'] == 'text':
            text = d['content']
            page_num = d['page_num']
            text = correct_line_breaks(text)
            dois.extend(extract_doi(text))
            text = remove_special_characters(text)
            if contain_keywords(text, key_words):
                return_data.append({
                    'text': text,
                    "page_num": page_num,
                })
        if d['type'] == 'table':
            pass
        if d['type'] == 'image':
            pass
    return_data = merge_contents(return_data, num_tokens_gpt_3_5 - count_tokens(prompt))
    if dois:
        doi = dois[0]
    else:
        doi = ''
    return return_data, file_name, doi


def pre_process_for_pypdf2(json_data, key_words):
    file_name = json_data['file_name']
    data = json_data['data']
    return_data = []
    all_text = []

    key_words = expend_keywords(key_words)
    for d in data:
        text = d['content']
        page_num = d['page_num']
        text = correct_line_breaks(text)
        text = remove_special_characters(text)
        all_text.append(text)

    all_text = "\n\n".join(all_text)
    all_text = remove_ref(all_text)
    splits = split_strings(all_text, num_tokens_gpt_3_5)
    splits = filter_by_keywords(splits, key_words)

    for split in splits:
        dois = extract_doi(split)
        return_data.append({
            'text': split,
            'num_tokens': count_tokens(split),
            'doi': dois[0] if dois else None  # 如果存在DOI，则添加到数据中，否则为None
        })

    return return_data, file_name


client = OpenAI(
    api_key="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",
)

def get_gpt_response(data):
    responses = []

    for item in data:
        text_input = item['text']
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt, },
                {"role": "user", "content": f"{text_input}", },
            ],
            model="gpt-3.5-turbo-1106",
            # model="gpt-4",
            temperature=0,
        )
        response = response.choices[0].message.content.strip()
        # print(response)
        responses.append(response)

    return responses

def parse_table(response_text, headers):
    lines = response_text.split('\n')
    table_lines = [line for line in lines if line.startswith('|') and line.endswith('|')]

    data_lines = table_lines[1:]
    data = []
    for line in data_lines:
        # 分割每个单元格并去除前后的空格
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if all(cell.strip().replace('-', '') == '' for cell in cells):
            continue
        if all(cell.strip().replace('N/A', '') == '' for cell in cells[2:]):
            continue
        # 处理 'N/A' 的情况
        # cells = [None if cell == 'N/A' else cell for cell in cells]
        # 确保数据行的长度与表头相匹配
        while len(cells) < len(headers):
            cells.append('N/A')
        data.append(cells)

    return pd.DataFrame(data, columns=headers)


def expand_rows(df, exclude_columns):
    # Initialize a list to hold the new rows
    new_rows = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the data for the excluded columns to be copied to new rows
        base_data = row[exclude_columns].to_dict()

        # Identify the columns with comma-separated values
        expandable_data = row.drop(labels=exclude_columns)

        # Determine the maximum number of splits across all expandable columns
        max_splits = max(
            [len(str(value).split(',')) for value in expandable_data if pd.notna(value) and ',' in str(value)] + [1])

        # Initialize a list to hold the expanded or split data for this row
        expanded_data = [base_data.copy() for _ in range(max_splits)]

        for col, value in expandable_data.items():
            if pd.notna(value) and ',' in str(value):
                # Split the value into a list, and ensure it covers all expanded rows
                split_values = [x.strip() for x in str(value).split(',')] + [np.nan] * (
                            max_splits - len(value.split(',')))

                for i, split_value in enumerate(split_values):
                    expanded_data[i][col] = split_value
            else:
                for expanded_row in expanded_data:
                    expanded_row[col] = value

        new_rows.extend(expanded_data)

    # Create a new DataFrame from the list of new rows
    expanded_df = pd.DataFrame(new_rows).reset_index(drop=True)
    return expanded_df

if __name__ == '__main__':
    input_dir = 'D:/ML/ysn_pdf/input_dir'
    cache_dir = 'D:/ML/ysn_pdf/cache_dir'
    failed_dir = 'D:/ML/ysn_pdf/failed_dir'
    output_dir = 'D:/ML/ysn_pdf/output_dir'
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    keywords = ['Reaction Type', 'Gibbs free energy',
                'GH*', 'GOH*', 'GOOH*', 'GO*', 'overpotential',
                'limiting potential', 'band gap', 'bandgap',
                'electron affinity', 'electronegativity',
                'Theoretical exchange current', 'charge transfer number']
    headers = ['Material Name', 'Reaction Type', 'Gibbs Free Energy',
               'ΔGO*', 'ΔGOH*', 'ΔGOOH*', 'ΔGH*',
               'Overpotential', 'Limiting Potential', 'Band Gap',
               'Electron Affinity', 'Electronegativity',
               'Theoretical Exchange Current', 'Charge Transfer Number']
    exclude_columns = ['Material Name', 'Reaction Type']
    json_files = get_cache_files(cache_dir)

    # for json_file in json_files:
    for json_file in tqdm(json_files):
        # print(os.path.join(cache_dir, json_file))
        json_file_path = os.path.join(cache_dir, json_file)
        csv_file_name = json_file[:-5] + '.csv'
        raw_file_name = json_file[:-5] + '.txt'
        output_file_path = os.path.join(output_dir, csv_file_name)
        raw_file_path = os.path.join(output_dir, raw_file_name)

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        data, file_name, doi = pre_process_for_3(data, keywords)
        # data, file_name = pre_process_for_pypdf2(data, keywords)

        responses = get_gpt_response(data)
        parsed_tables = [parse_table(response, headers) for response in responses]

        all_data = []
        counter = 1
        failed_pdfs = []
        try:
            for item, table in zip(data, parsed_tables):
                table = expand_rows(table, exclude_columns)
                table['file_name'] = [file_name] * len(table)
                table['text'] = [item['text']] * len(table)
                if 'page_num' in item:
                    table['page_num'] = [item['page_num']] * len(table)
                table['content_block'] = [counter] * len(table)
                table['num_tokens'] = [item['num_tokens']] * len(table)
                table['doi'] = [doi] * len(table)
                all_data.append(table)
                counter += 1

            df = pd.concat(all_data, ignore_index=True)
            if 'page_num' in item:
                new_column_order = ['file_name', 'doi', 'content_block', 'page_num', 'text', 'num_tokens'] + headers
            else:
                new_column_order = ['file_name', 'doi', 'content_block', 'text', 'num_tokens'] + headers
            df = df[new_column_order]
        except:
            base_info_data = {
                'file_name': [file_name],
                'text': [item['text']],
                'page_num': [item.get('page_num', None)],
                'content_block': [counter],
                'num_tokens': [item['num_tokens']],
                'doi': [doi]
            }
            table = pd.DataFrame(base_info_data)

            for header in headers:
                if header not in table:
                    table[header] = "N/A"
            if 'page_num' in item:
                new_column_order = ['file_name', 'doi', 'content_block', 'page_num', 'text', 'num_tokens'] + headers
            else:
                new_column_order = ['file_name', 'doi', 'content_block', 'text', 'num_tokens'] + headers
            df = table[new_column_order]
            failed_pdfs.append(json_file_path)

        df.to_csv(output_file_path, index=False, encoding='utf-8', escapechar='\\')

        with open(raw_file_path, 'w', newline='', encoding='utf-8') as f:
            for response in responses:
                f.write(response + '\n\n--\n\n')

    save_failed_pdfs(failed_dir, failed_dir)