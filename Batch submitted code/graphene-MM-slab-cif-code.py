from itertools import combinations_with_replacement
from ase.build import graphene
from ase.io import write
from ase import Atoms
import os

def substitute_carbon_with_nitrogen(atoms, carbon_indices, num_nitrogen):
    # Substitute specified number of carbon atoms with nitrogen atoms
    for i in range(num_nitrogen):
        if i < len(carbon_indices):
            atoms[carbon_indices[i]].symbol = 'N'

def create_graphene_with_metal_and_nitrogen_combinations(metals, positions, carbon_indices, special_folder):
    special_metals = {'Fe', 'Co', 'Ni', 'Cr', 'Mn'}  # 特定的金属元素集合
    for combo in combinations_with_replacement(metals, 2):
        for num_nitrogen in range(len(carbon_indices) + 1):
            # Create a single-layer graphene sheet and a 5x5 supercell
            graphene_sheet = graphene()
            large_graphene_sheet = graphene_sheet.repeat((5, 5, 1))
            large_graphene_sheet.center(vacuum=10, axis=2)

            # Indices of the carbon atoms to be removed
            indices_to_remove = [24, 26, 15, 17]
            modified_graphene_sheet = large_graphene_sheet.copy()
            del modified_graphene_sheet[indices_to_remove]

            # Add metal atoms from the combination
            for metal, position in zip(combo, positions):
                metal_atom = Atoms([metal], positions=[position])
                modified_graphene_sheet.extend(metal_atom)

            # Substitute specified carbon atoms with nitrogen atoms
            substitute_carbon_with_nitrogen(modified_graphene_sheet, carbon_indices, num_nitrogen)

            # Check if the combination contains any special metals
            if any(metal in special_metals for metal in combo):
                output_folder = special_folder
            else:
                output_folder = '.'  # Current directory

            # Save the file to the appropriate folder
            modified_filename = os.path.join(output_folder, f'{combo[0]}-{combo[1]}-{num_nitrogen}N.cif')
            write(modified_filename, modified_graphene_sheet)
            print(f"Modified graphene structure with {combo[0]}, {combo[1]} and {num_nitrogen} N substitutions saved as: {modified_filename}")

# Metals, their positions, and specific carbon atom indices
metals = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Sc', 'Ti', 'V', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
positions = [(0.71, 6.79, 10), (1.77, 4.55, 10)]
carbon_indices = [24, 15, 23, 14, 13, 22]  # Example indices of carbon atoms to be substituted

special_folder = 'cixing-group'  # Folder for special combinations
if not os.path.exists(special_folder):
    os.makedirs(special_folder)

create_graphene_with_metal_and_nitrogen_combinations(metals, positions, carbon_indices, special_folder)
