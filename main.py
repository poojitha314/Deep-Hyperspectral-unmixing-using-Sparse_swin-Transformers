# import random
# import torch
# import numpy as np
# from Trans_mod import Train_test

# # Set seed for reproducibility
# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

# # Device Configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("\nSelected device:", device, end="\n\n")

# # Initialize and run the training/testing module
# tmod = Train_test(dataset='samson', device=device, skip_train=False, save=True)
# tmod.run(smry=False)






# import random
# import torch
# import numpy as np
# from Trans_mod import Train_test

# # Set seed for reproducibility
# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

# # Select device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("Selected device:", device, "\n")

# # Run SparseSwin model for all datasets
# datasets = ['samson', 'apex','jasperRidge', 'Urban']

# for ds in datasets:
#     print(f"\n=== Running SparseSwin model on {ds.upper()} dataset ===\n")
#     tmod = Train_test(dataset=ds, device=device, skip_train=False, save=True)
#     tmod.run(smry=False)



import random
import torch
import numpy as np
from Trans_mod import Train_test

# Seed for reproducibility
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Selected device:", device, "\n")

datasets = ['samson', 'apex', 'jasperRidge', 'Urban']


for ds in datasets:
    print(f"\n=== Running SparseSwin model on {ds.upper()} dataset ===\n")
    tmod = Train_test(dataset=ds, device=device, skip_train=False, save=True)
    tmod.run(smry=False)
