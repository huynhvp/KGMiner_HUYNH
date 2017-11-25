file1_names = open('./data_id/state_largest_cities.csv', 'r')
ids = open('./data_id/state_largest_cities_id.csv', 'w')

ids.write('state_id,city_id\n')

file1_name = file1_names.readline()
state_name = file1_name.split(',')[0]
city_name = file1_name.split(',')[1].split('\n')[0]

while file1_name != "":
    file1_found = False
    file2_found = False
    str_to_write = ","
    nodes = open('./data/infobox.nodes', 'r')
    node = nodes.readline()
    while node != "" and (not file1_found or not file2_found):
        node_info = node.split('\t')
        if node_info[1].split('\n')[0] == city_name+',_'+state_name:
            file2_found = True
            str_to_write = str_to_write + node_info[0]
        if node_info[1].split('\n')[0] == state_name:
            file1_found = True
            str_to_write = node_info[0] + str_to_write
        node = nodes.readline()
    if (not file2_found):
        nodes = open('/home/phi/KGMiner/data/infobox.nodes', 'r')
        node = nodes.readline()
        while node != "" and (not file2_found):
            node_info = node.split('\t')
            if node_info[1].split('\n')[0] == city_name:
                file2_found = True
                str_to_write = str_to_write + node_info[0]
        
    print "writing " + str_to_write + " meanings: " + state_name + "," + city_name
    ids.write(str_to_write + '\n')
    file1_name = file1_names.readline()
    state_name = file1_name.split(',')[0]
    city_name = file1_name.split(',')[1].split('\n')[0]
ids.close()
