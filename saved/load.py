import pickle
import matplotlib.pyplot as plt

epoch_loss_lst, epoch_error_lst, epoch_error_lst_val = pickle.load( open( "savedVals.askoy.esat.kuleuven.be.pkl", "rb" ) )


plt.plot(epoch_loss_lst)
plt.plot(epoch_error_lst)
plt.plot(epoch_error_lst_val)
plt.show()
