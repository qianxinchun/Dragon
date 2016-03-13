set DATA_DIR=G:/Dataset/cifar10/DATA
set DB_DIR=G:/Dataset/cifar10
start /wait convert_cifar.exe %DATA_DIR%  %DB_DIR% lmdb
start compute_mean.exe -backend=lmdb %DB_DIR%/cifar10_train_lmdb  %DB_DIR%/mean.binary