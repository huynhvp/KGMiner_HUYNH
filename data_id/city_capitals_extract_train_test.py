file1_names = open('./city_capitals/city_capitals_label.tsv', 'r')
training_file = open('./city_capitals/training.tsv', 'w')
testing_file = open('./city_capitals/testing.tsv', 'w')

file1_name = file1_names.readlines()

print file1_name

kf = KFold(n_splits=10, shuffle = True, random_state=233)
i_fold = 1;
for i_train, i_test in kf.split(file1_name):
    training_name = './city_capitals/training_' + str(i_fold) + '.tsv'
    testing_name = './city_capitals/testing_' + str(i_fold) + '.tsv'

    train_set = file1_name[i_train]
    test_set = file1_name[i_test]

    training_file = open(training_name, 'w')
    testing_file = open(testing_name, 'w')

    for i in range(len(training_set)):
        training_file.write(training_set[i])

    for i in range(len(testing_set)):
        testing_file.write(testing_set[i])

    training_file.close()
    testing_file.close()

    i_fold+=1


