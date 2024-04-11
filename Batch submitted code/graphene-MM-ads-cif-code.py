
from itertools import permutations
from ase.build import graphene
from ase.io import write
from ase import Atoms

def substitute_carbon_with_nitrogen(atoms, carbon_indices, num_nitrogen):
    for i in range(num_nitrogen):
        if i < len(carbon_indices):
            atoms[carbon_indices[i]].symbol = 'N'

def create_graphene_with_single_hydrogen(metals, positions, carbon_indices):
    processed_combinations = set()  # 用于跟踪已处理的金属组合和氢原子位置

    # 循环遍历所有金属组合
    for combo in permutations(metals, 2):
        # 循环遍历所有可能的氮原子替换数量
        for num_nitrogen in range(len(carbon_indices) + 1):
            graphene_sheet = graphene()
            large_graphene_sheet = graphene_sheet.repeat((5, 5, 1))
            large_graphene_sheet.center(vacuum=10, axis=2)
            indices_to_remove = [24, 26, 15, 17]
            modified_graphene_sheet = large_graphene_sheet.copy()
            del modified_graphene_sheet[indices_to_remove]

            # 添加金属原子
            for metal, position in zip(combo, positions):
                metal_atom = Atoms([metal], positions=[position])
                modified_graphene_sheet.extend(metal_atom)

            # 替换指定的碳原子为氮原子
            substitute_carbon_with_nitrogen(modified_graphene_sheet, carbon_indices, num_nitrogen)

            # 添加一个氧原子，并为每个金属原子位置生成结构
            metal_pos = positions[0]  # 只在第一个金属原子上添加氧原子
            oxygen_position = (metal_pos[0], metal_pos[1], metal_pos[2] + 2.0)
            structure_with_oxygen = modified_graphene_sheet.copy()
            oxygen_atom = Atoms(['O'], positions=[oxygen_position])
            structure_with_oxygen.extend(oxygen_atom)

            # 生成文件名
            modified_filename = f'{combo[0]}-{combo[1]}-{num_nitrogen}N-O.cif'
            write(modified_filename, structure_with_oxygen)
            print(f"Structure saved as: {modified_filename}")

            # 如果金属相同，再生成一次，交换顺序
            if combo[0] == combo[1]:
                structure_with_oxygen = modified_graphene_sheet.copy()
                modified_filename = f'{combo[1]}-{combo[0]}-{num_nitrogen}N-O.cif'
                write(modified_filename, structure_with_oxygen)
                print(f"Structure saved as: {modified_filename}")

    # 处理金属相同的情况
    for metal in metals:
        combo = (metal, metal)
        for num_nitrogen in range(len(carbon_indices) + 1):
            graphene_sheet = graphene()
            large_graphene_sheet = graphene_sheet.repeat((5, 5, 1))
            large_graphene_sheet.center(vacuum=10, axis=2)
            indices_to_remove = [24, 26, 15, 17]
            modified_graphene_sheet = large_graphene_sheet.copy()
            del modified_graphene_sheet[indices_to_remove]

            # 添加金属原子
            for position in positions:
                metal_atom = Atoms([metal], positions=[position])
                modified_graphene_sheet.extend(metal_atom)

            # 替换指定的碳原子为氮原子
            substitute_carbon_with_nitrogen(modified_graphene_sheet, carbon_indices, num_nitrogen)

            # 添加一个氧原子，并为每个金属原子位置生成结构
            metal_pos = positions[0]  # 只在第一个金属原子上添加氧原子
            oxygen_position = (metal_pos[0], metal_pos[1], metal_pos[2] + 2.0)
            structure_with_oxygen = modified_graphene_sheet.copy()
            oxygen_atom = Atoms(['O'], positions=[oxygen_position])
            structure_with_oxygen.extend(oxygen_atom)

            # 生成文件名
            modified_filename = f'{metal}-{metal}-{num_nitrogen}N-O.cif'
            write(modified_filename, structure_with_oxygen)
            print(f"Structure saved as: {modified_filename}")

# 定义金属、位置和碳原子索引
metals = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Sc', 'Ti', 'V', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag',
          'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
positions = [(0.71, 6.79, 10), (1.77, 4.55, 10)]
carbon_indices = [24, 15, 23, 14, 13, 22]
create_graphene_with_single_hydrogen(metals, positions, carbon_indices)
