file1_names = open('./data_id/bestseller.csv', 'r')
ids = open('./data_id/NYT_bestseller.tsv', 'w')


file1_name = file1_names.readline()
file1_name = file1_names.readline()
state_id = file1_name.split(',')[0]
city_id = file1_name.split(',')[1]
is_capital = file1_name.split(',')[2].split('\n')[0]

f = open('./data/infobox.nodes', 'r')
nodes = f.readlines()
node_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
from collections import OrderedDict
vertexmap = OrderedDict(( (i,entity) for i, entity in sorted(node_dict) ))
del node_dict
del nodes

while file1_name != "":
    str_to_write = vertexmap[state_id] +  "\t" + vertexmap[city_id]
    str_to_write += "\t" + ("1" if is_capital=="TRUE" else "-1")
    print "writing " + str_to_write 
    ids.write(str_to_write + '\n')
    file1_name = file1_names.readline()
    state_id = file1_name.split(',')[0]
    city_id = file1_name.split(',')[1]
    is_capital = file1_name.split(',')[2].split('\n')[0]
ids.close()
