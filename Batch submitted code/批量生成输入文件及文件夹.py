import os
from ase.io import read, write
from collections import OrderedDict
from shutil import copyfile
from ase import Atoms

def create_potcar(atoms, potcar_dir, dest_folder, alternative_potcar=None):
    # 使用 sort_atoms_by_type 函数返回的元素顺序
    element_order = [atom.symbol for atom in atoms]
    unique_elements_ordered = OrderedDict.fromkeys(element_order)

    potcar_content = ''
    for element in unique_elements_ordered:
        potcar_path = os.path.join(potcar_dir, element, 'POTCAR')
        if not os.path.exists(potcar_path):
            if alternative_potcar and element in alternative_potcar:
                alt_potcar_path = os.path.join(potcar_dir, alternative_potcar[element], 'POTCAR')
                if not os.path.exists(alt_potcar_path):
                    raise ValueError(f'Neither standard nor alternative POTCAR found for {element}')
                potcar_path = alt_potcar_path
            else:
                raise ValueError(f'POTCAR for {element} not found in {potcar_path}')

        with open(potcar_path, 'r') as file:
            potcar_content += file.read()

    with open(os.path.join(dest_folder, 'POTCAR'), 'w') as file:
        file.write(potcar_content)

def sort_atoms_by_type(atoms):
    elements_order = OrderedDict()
    for atom in atoms:
        elements_order.setdefault(atom.symbol, []).append(atom)

    # 使用提取的原子信息创建一个新的 Atoms 对象，并包含原始晶格信息
    sorted_atoms = Atoms([atom for symbol in elements_order for atom in elements_order[symbol]],
                         cell=atoms.cell, pbc=atoms.pbc)
    return sorted_atoms

# 路径配置
structure_directory = 'D:/work/gre/MM/ads/cixing-cif'
vasp_input_files = ['INCAR', 'KPOINTS', 'vasp.pbs']
output_directory = 'D:/work/gre/MM/ads/cixing-out'
potcar_dir = 'D:/work/gre/pbe54'
alternative_potcar = {'Nb': 'Nb_sv', 'Y': 'Y_sv', 'Zr': 'Zr_sv'}

# 处理结构文件
for structure_file in os.listdir(structure_directory):
    if structure_file.endswith('.cif'):
        structure = read(os.path.join(structure_directory, structure_file))

        sorted_structure = sort_atoms_by_type(structure)

        folder_name = structure_file.split('.')[0]
        folder_path = os.path.join(output_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        write(os.path.join(folder_path, 'POSCAR'), sorted_structure)

        create_potcar(sorted_structure, potcar_dir, folder_path, alternative_potcar) # 使用排序后的结构来创建 POTCAR

        for file in vasp_input_files:
            src_file = os.path.join('D:/work/gre/MM/ads/cixing-input', file)
            dest_file = os.path.join(folder_path, file)
            copyfile(src_file, dest_file)

        print(f"Folder created for structure: {folder_name}")

print("All folders created.")
