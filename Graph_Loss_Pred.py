#import Statements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
         ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))

    b = np.mean(ys) - m * np.mean(xs)

    return m, b



def dataplt(dem,color,title):
    Actual = [None]*6
    Prediction = [None]*6
    for i in range(6):
        Actual[i] = np.loadtxt('Graph_Data/' + dem + '_G_Data/Actual_' + str(i) + ' ' + dem, delimiter=',')
        Prediction[i] = np.loadtxt('Graph_Data/'+dem+'_G_Data/Prediction_'+str(i)+' '+dem, delimiter=',')
    Test_Loss = np.loadtxt('Graph_Data/'+dem+'_G_Data/Test Loss '+dem)
    Train_Loss = np.loadtxt('Graph_Data/'+dem+'_G_Data/Training Loss '+dem)
    Validation_Loss = np.loadtxt('Graph_Data/'+dem+'_G_Data/Validation Loss '+dem)
    Test_coordinate = [(i+1) for i in range(len(Test_Loss))]
    Validation_coordinate = [(i+1) for i in range(len(Validation_Loss))]
    Train_coordinate = [(i+1) for i in range(len(Train_Loss))]
    f = plt.figure(figsize=(20, 10))
    ax = [None]*6
    ax[0] = f.add_subplot(231)
    ax[1] = f.add_subplot(232)
    ax[2] = f.add_subplot(233)
    ax[3] = f.add_subplot(234)
    ax[4] = f.add_subplot(235)
    ax[5] = f.add_subplot(236)
    x = np.linspace(0, 1, 10)
    for i in range(6):
        ax[i].plot(x, x + 0, linestyle='solid',linewidth=3, color='lime')
        ax[i].scatter(Prediction[i], Actual[i], c =color)
        m,b = best_fit_slope_and_intercept(Prediction[i],Actual[i])
        ax[i].plot(x,m*x+b,linestyle='-.',linewidth=3, color='cyan')
        ax[i].set(xlabel='Predicted')
        ax[i].set(ylabel='Actual')
        ax[i].legend(['x=y', 'Line of Best Fit'], loc=4)
        print(str(m)+'x'+str(b)+' '+str(i)+title)
    ax[0].set_title('Predicted vs Actual - Omega Matter - '+title)
    ax[1].set_title('Predicted vs Actual - σ_8 - ' + title)
    ax[2].set_title('Predicted vs Actual - a_1 - ' + title)
    ax[3].set_title('Predicted vs Actual - a_2 - ' + title)
    ax[4].set_title('Predicted vs Actual - a_3 - ' + title)
    ax[5].set_title('Predicted vs Actual - a_4 - ' + title)
    plt.savefig('Predicted vs Actual - ' + title, dpi=300, bbox_inches='tight')
    plt.show()
    plt.semilogy(Train_coordinate,Train_Loss)
    plt.title('Train Loss - ' +title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Train Loss - ' +title, dpi=300, bbox_inches='tight')
    plt.show()
    plt.semilogy(Validation_coordinate, Validation_Loss)
    plt.title('Validation Loss - ' + title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Validation Loss - ' + title, dpi=300, bbox_inches='tight')
    plt.show()
    plt.semilogy(Test_coordinate, Test_Loss)
    plt.title('Test Loss - ' + title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Test Loss - ' + title, dpi=300, bbox_inches='tight')
    plt.show()
    return Test_Loss[99]
Categories = ['T', 'P Gas', 'Metal', 'S Mass', 'G ρ', 'E ρ',
              'DM ρ', 'G Vel.','Neutral H']
losses = [1,2,3,4,5,6,7,8,9]
losses[0] = dataplt(dem='T',color = 'red', title ='Temperature')
losses[1] = dataplt(dem='P',color = 'green', title ='Gas Pressure')
losses[2] = dataplt(dem='Z',color = 'black', title ='Metallicity')
losses[3] = dataplt(dem='Mstar',color = 'blue', title ='Stellar Mass')
losses[4] = dataplt(dem='Mgas',color = 'olive', title ='Gas Density')
losses[5] = dataplt(dem='ne',color = 'orchid', title ='Electron Density')
losses[6] = dataplt(dem='Mcdm',color = 'indigo', title ='Dark Matter Density')
losses[7] = dataplt(dem='Vgas',color = 'teal', title ='Gas Velocity')
losses[8] = dataplt(dem='HI',color = 'cyan', title ='Neutral Hydrogen')


# The first parameter would be the x value,
# by editing the delta between the x-values
# you change the space between bars
plt.bar(Categories, losses)

plt.show()
print('Exit')

