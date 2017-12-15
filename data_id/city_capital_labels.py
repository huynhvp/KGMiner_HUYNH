file1_names = open('./data_id/city_capital.csv', 'r')
ids = open('./data_id/city_capitals_labels.tsv', 'w')


file1_name = file1_names.readline()
file1_name = file1_names.readline()
state_id = file1_name.split(',')[0]
city_id = file1_name.split(',')[1]
is_capital = file1_name.split(',')[2].split('\n')[0]

while file1_name != "":
    file1_found = False
    file2_found = False
    str_to_write = "\t"
    nodes = open('./data/infobox.nodes', 'r')
    node = nodes.readline()
    while node != "" and (not file1_found or not file2_found):
        node_info = node.split('\t')
        if node_info[0] == city_id:
            file2_found = True
            str_to_write = str_to_write + node_info[1].split('\n')[0]
        if node_info[0] == state_id:
            file1_found = True
            str_to_write = node_info[1].split('\n')[0] + str_to_write
        node = nodes.readline()
    str_to_write += "\t" + ("1" if is_capital=="TRUE" else "-1")
    print "writing " + str_to_write 
    ids.write(str_to_write + '\n')
    file1_name = file1_names.readline()
    state_id = file1_name.split(',')[0]
    city_id = file1_name.split(',')[1]
    is_capital = file1_name.split(',')[2].split('\n')[0]
ids.close()
