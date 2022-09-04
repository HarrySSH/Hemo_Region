from matplotlib.pyplot import figure
import  matplotlib.pyplot as plt
import seaborn as sns
def encoder(celltype = None):
    dic={"B1":0, "B2":0,'E0':0, "E1":0,'E2':0,'E3':0, "E4":0, "ER1":0, "ER2":0,
                             "ER3":0, "ER4":0, "ER5":0,
                             "ER6":0, "L2":0, "L4":0,
                            "M1":0,
                            "M2":0, "M3":0, "M4":0,
                            "M5":0, "M6":0, 
         'MO1':0,"MO2":0,
        'PL2':0,'PL3':0,'PL4':0,
        'U1':0,'U4':0}
    p = 0
    for _celltype in celltype:
        dic[_celltype] = 15 + p
        p += 5
    return dic

def plot_UMAP(projections, cellnames_test, string = None, subset = False):
    figure(figsize=(16, 12), dpi=80)
    x = projections[:, 0]
    y = projections[:, 1]

    classes = cellnames_test['high_level'].tolist()
    if subset == True:
        
        colours = [sns.color_palette("ch:s=-.2,r=.6",50)[x] for x in cellnames_test['high_level'].map(encoder(celltype=string))]
        indices = [x for x in cellnames_test['high_level'].map(encoder(celltype=string))]
        classes = ['Others' if y==0 else x for x, y in zip(classes, indices) ]
    else:
        celltypes = list(set(cellnames_test['high_level']))
        celltypes.sort()
        '''
        maps = {}
        length = len(celltypes)
        _k = 0
        for _cell in celltypes:
            maps[_cell] = _k
            _k +=1
        '''
        maps={"B1":0, "B2":0,
             'E0':1, "E1":1,'E2':1,'E3':1, "E4":1, 
             "ER1":2, "ER2":2, "ER3":2, "ER4":2, "ER5":2,"ER6":2, 
             "L2":3, "L4":3,
             "M1":4, "M2":4, "M3":4, "M4":4, "M5":4, "M6":4, 
         'MO1':5,"MO2":5,
        'PL2':6,'PL3':6,'PL4':6,
        'U1':7,'U4':7}
        
        colours = [sns.color_palette("hls", 8)[x] for x in cellnames_test['high_level'].map(maps)]
    
    for (i,cla) in enumerate(set(classes)):
        xc = [p for (j,p) in enumerate(x) if classes[j]==cla]
        yc = [p for (j,p) in enumerate(y) if classes[j]==cla]
        cols = [c for (j,c) in enumerate(colours) if classes[j]==cla]
        plt.scatter(xc,yc,c=cols)
    #plt.legend(loc=4)
    plt.axis('off')