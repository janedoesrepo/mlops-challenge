This readme will not explain your task but will only cover what already is there in the code. 

The main tasks performed in this code are:
 - Preprocess/transform data to the format that can be consumed by the model;
 - Implement UNet to detect building footprints;
 - Train and evaluate the model's results

In this project you will find:

    -requirements.txt it contains all the necessary libraries;
    -scripts contains a modular code;
    -trained_models contains the best model based on the dice score and the last trained model;

------------------------
Train segmentation model
------------------------
python ./LiveEO_MLOps_Challenge/scripts/main.py --base_dir "test_images" --num_epochs 10 --exec_mode 'train'

-----------------------
Test segmentation model
-----------------------

python ./LiveEO_MLOps_Challenge/scripts/main.py --base_dir "test_images" --exec_mode 'evaluate' --ckpt_path './last.ckpt'


-----------------------------
Load and display some samples
-----------------------------

preds = np.load('./predictions.npy')   #(6, 1, 1024, 1024)
lbls = np.load('./labels.npy')         #(6, 1, 1024, 1024)

# plot some examples
fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].imshow(preds[3][0], cmap='gray')
ax[1].imshow(lbls[3][0], cmap='gray')