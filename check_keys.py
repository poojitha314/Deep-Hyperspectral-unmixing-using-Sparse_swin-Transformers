import scipy.io as sio

# Load the .mat files
data_sam = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/samson_dataset.mat")
data_ap = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/apex_dataset.mat")
data_dc = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/dc_dataset.mat")
data_jsr = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/jasperRidge_dataset.mat")
data_ur = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/Urban_dataset.mat")

# data_jsr_gt = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/jasperRidge_GT.mat")
data_ur_gt = sio.loadmat("C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/end4_groundTruth.mat")

# Print keys for each dataset
print("Keys found in samson_dataset.mat:")
for key in data_sam.keys():
    print(key)

print("\nKeys found in apex_dataset.mat:")
for key in data_ap.keys():
    print(key)

print("\nKeys found in dc_dataset.mat:")
for key in data_dc.keys():
    print(key)

# print("\nKeys found in jasperRidge_GT_dataset.mat:")
# for key in data_jsr_gt.keys():
#     print(key)

print("\nKeys found in Urban_GT_dataset2.mat:")
for key in data_ur_gt.keys():
    print(key)

print("\nKeys found in jasperRidge_dataset.mat:")
for key in data_jsr.keys():
    print(key)

print("\nKeys found in Urban_dataset.mat:")
for key in data_ur.keys():
    print(key)
