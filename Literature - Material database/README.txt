1. 如果配置代理使用gpt ，在llm_parser.py设置IP和端口。不需要的话请删除这两行，否则网络不通
os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"

2. pdf_parser.py需要设置输入目录和输出目录input_dir，cache_dir，failed_dir

3. llm_parser.py需要设置输入目录和输出目录cache_dir，output_dir

4. 用到的库pip install openai tiktoken PyPDF2 pdfminer.six pdfplumber pdf2image Pillow pytesseract pandas nltk tqdm
如果用的是清华的镜像则是下边的版本：
pip install openai tiktoken PyPDF2 pdfminer.six pdfplumber pdf2image Pillow pytesseract pandas nltk tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

5. nltk库的配置
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
http://www.nltk.org/nltk_data/
  Searched in:
    - 'C:\\Users\\liang/nltk_data'
    - 'D:\\workspace\\LLM\\LLM\\llm_pdf\\nltk_data'
    - 'D:\\workspace\\LLM\\LLM\\llm_pdf\\share\\nltk_data'
    - 'D:\\workspace\\LLM\\LLM\\llm_pdf\\lib\\nltk_data'
    - 'C:\\Users\\liang\\AppData\\Roaming\\nltk_data'
    - 'C:\\nltk_data'
    - 'D:\\nltk_data'
    - 'E:\\nltk_data'
ysn
activate
import nltk
nltk.find('.')
  Searched in:
    - 'C:\\Users\\yangshuna/nltk_data'
    - 'F:\\installl\\Anaconda3-2023\\nltk_data'
    - 'F:\\installl\\Anaconda3-2023\\share\\nltk_data'
    - 'F:\\installl\\Anaconda3-2023\\lib\\nltk_data'
    - 'C:\\Users\\yangshuna\\AppData\\Roaming\\nltk_data'
    - 'C:\\nltk_data'
    - 'D:\\nltk_data'
    - 'E:\\nltk_data'



6. 选择pdf的方案请llm_parser.py中，在下面两行中选择一个（可以搜索）
        data, file_name = pre_process_for_3(data, keywords)
        # data, file_name = pre_process_for_pypdf2(data, keywords)
pdf_parser.py，在下面两行中选择一个
            data = pdf_to_txt_3(os.path.join(input_dir, pdf_file))
            # data = pdf_to_txt_pypdf2(os.path.join(input_dir, pdf_file))

7. 修改提示词需要同时修改下面的内容。
    keywords = ['Reaction Type', 'Gibbs free energy',
                'GH*', 'GOH*', 'GOOH*', 'GO*', 'overpotential', 
                'limiting potential', 'band gap', 'bandgap',
                'electron affinity', 'electronegativity',
                'Theoretical exchange current', 'charge transfer number']
    headers = ['Material Name','Reaction Type','Gibbs Free Energy',
            'ΔGO*', 'ΔGOH*', 'ΔGOOH*', 'ΔGH*',
            'Overpotential','Limiting Potential','Band Gap',
            'Electron Affinity','Electronegativity',
            'Theoretical Exchange Current','Charge Transfer Number']
prompt = """
You will be provided with text containing detailed information on various materials used in chemical reactions.
Your task is to carefully read the text and extract specific information about each material.

For each material, create a table with the following columns:
Material Name: compound name or chemical formula, e.g. V-CoOOH/CoS2, Ni@BN-yne, Pt doped GaPS4.
Reaction Type: type of reaction, such as Oxygen Reduction Reaction(ORR), Hydrogen Evolution Reaction(HER).
Gibbs free energy (ΔGH*, ΔGOH*, ΔGOOH*, ΔGH*): or adsorption energy, the unit is eV.
ΔGO*: gibbs free energy of adsorption O or *O.
ΔGOH*: gibbs free energy of adsorption OH or *OH.
ΔGOOH*: gibbs free energy of adsorption OOH or *OOH.
ΔGH*: gibbs free energy of adsorption H or *H.
Overpotential: abbreviation \eta, the unit is V, eV or mV, it refers to the part of the electrocatalytic reaction that the actual voltage required to reach a certain current density exceeds the theoretical voltage. e.g. 0.97 eV.
Limiting Potential: abbreviation UL, the unit is V, eV or mV. The limit value of the potential difference in the battery reaction. e.g. 1.51 eV.
Electronegativity: A relative scale of the ability of atoms to attract electrons of each element in the periodic table. e.g. N (Nitrogen): 3.04.
Band gap: or bandgap, e.g. 2.90 eV.
Electron affinity: the ability of an atom or molecule to donate or receive electrons to the outside.
Theoretical exchange current: The electrode reaction is in equilibrium, and the current density corresponding to the cathode and anode can describe the ability of the electrode to gain and lose electrons in the reaction, and the symbol is io and o is the subscript.
Charge Transfer Number: The number of electrons gained and lost in a chemical reaction, is generally expressed by n.

If you find the relevant information, present it in a tabular format like this:
| Material Name | Reaction Type | Gibbs Free Energy | ΔGO* | ΔGOH* | ΔGOOH* | ΔGH* | Overpotential | Limiting Potential | Band Gap | Electron Affinity | Electronegativity | Theoretical Exchange Current | Charge Transfer Number |
| N-doped hierarchical Co3O4 (3D Co3O4/NC-250) | Acidic OER | 1.40 eV | 225 mV | -2.94 eV | 1.19 V | 1.51 V | 2.90 eV | O: 3.44 | N/A | N/A | N/A |

If you can not find any of the relevant data, just answer 'N/A' like this:
N/A
"""

8. 修改每块文本的长度修改下面内容。
prompt_tokens = count_tokens(prompt)
num_tokens_gpt_3_5 = 3000