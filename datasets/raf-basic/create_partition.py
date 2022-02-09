import pandas as pd


df = pd.read_csv('Annotation/manual/train_04602_manu_attri.txt',sep='\t')

df = pd.read_csv('EmoLabel/list_patition_label.txt', sep=' ', header=None)
file_names = df.iloc[:, 0].values
print(file_names[0])


type_list = {'GenderLabel'}

f4 = open("GenderLabel/list_patition_label.txt", "w")
f5 = open("RaceLabel/list_patition_label.txt", "w")
f6 = open("AgeLabel/list_patition_label.txt", "w")


for f in file_names:
    f0 = f.split(".")[0]
#    f = f0 +"_manu_attri.txt"
#    df = pd.read_csv('Annotation/manual/{}'.format(f),sep='\t')
#    f4.write('{:s}.jpg\t{:d}\n'.format(f0, int(df.iloc[4,0])))
#    f5.write('{:s}.jpg\t{:d}\n'.format(f0, int(df.iloc[5,0])))
#    f6.write('{:s}.jpg\t{:d}\n'.format(f0, int(df.iloc[6,0])))

f4.close()
f5.close()
f6.close()

