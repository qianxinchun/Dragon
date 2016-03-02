#pragma warning(disable:4244)
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <stdio.h>
#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"
#include "cstdlib"
#include "dragon.pb.h"
#include "db.hpp"

using std::string;
using namespace boost;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, unsigned char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read((char*)buffer, kCIFARImageNBytes);
  for (int i = 0; i < 1024; i++) cout << buffer[i] << endl;
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type) {
  scoped_ptr<DB> train_db(GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, DB::NEW);
  scoped_ptr<Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    _snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
		read_image(&data_file, &label, (unsigned char*)str_buffer);
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
	  // const string& data = datum.data();
	  //for (int j = 0; j < data.size(); j++) cout << (int)data[j] << endl;
      int length = _snprintf(str_buffer, kCIFARImageNBytes, "%05d",
          fileid * kCIFARBatchSize + itemid);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(string(str_buffer, length), out);
    }
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<DB> test_db(GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, DB::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
	  read_image(&data_file, &label, (unsigned char*)str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    int length = _snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(str_buffer, length), out);
  }
  txn->Commit();
  test_db->Close();
}
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  std::system("pause");
  return 0;
}