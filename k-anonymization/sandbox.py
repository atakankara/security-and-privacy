
lookup_table = {}

lines = open(r"./DGHs/workclass.txt").read().split("\n")

def create_DGH_tree(lines): 
    def _recursive_DGH_tree(lines, parent, first_index, last_index):
        if first_index + 1 >= last_index:
            return
        
        current_parent_index=0
        next_parent_index = 0
        for i in range(first_index, last_index):
            if lines[i].startswith('\t'):
                lines[i] = lines[i].replace('\t', '', 1)
            
            if not lines[i].startswith('\t'):
                lookup_table[lines[i].rstrip("\t")] = parent
                if current_parent_index == 0:
                    current_parent_index=i
                else:
                    _recursive_DGH_tree(lines, lines[current_parent_index], current_parent_index+1, i)
                    current_parent_index=i
            
        
        _recursive_DGH_tree(lines, lines[current_parent_index], current_parent_index+1, last_index)
                    

    _recursive_DGH_tree(lines, lines[0], 1, len(lines))

create_DGH_tree(lines)
print(lookup_table)