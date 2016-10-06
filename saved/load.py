import pickle
import matplotlib.pyplot as plt

epoch_loss_lst, epoch_error_lst, epoch_error_lst_val, test_err = pickle.load(open( "adam0510.pkl", "rb" ))


plt.plot(epoch_error_lst)
plt.plot(epoch_error_lst_val)
plt.show()
