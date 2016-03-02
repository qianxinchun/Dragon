#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "dragon.pb.h"
#include "db.hpp"
#include "io.hpp"

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb","The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
	google::InitGoogleLogging(argv[0]);
	google::LogToStderr();
	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	scoped_ptr<DB> db(GetDB(FLAGS_backend));
	db->Open(argv[1], DB::READ);
	scoped_ptr<Cursor> cursor(db->NewCursor());
	BlobProto sum_blob;
	int count = 0;
	// try to read first datum to reshape
	Datum datum;
	datum.ParseFromString(cursor->value());
	sum_blob.mutable_shape()->add_dim(1);
	sum_blob.mutable_shape()->add_dim(datum.channels());
	sum_blob.mutable_shape()->add_dim(datum.height());
	sum_blob.mutable_shape()->add_dim(datum.width());
	const int data_size = datum.channels() * datum.height() * datum.width();
	int size_in_datum = max<int>(datum.data().size(),datum.float_data_size());
	for (int i = 0; i < size_in_datum; ++i) sum_blob.add_data(0.);
	LOG(INFO) << "Compute mean: " << argv[1]<<endl;
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		const string& data = datum.data();
		size_in_datum = std::max<int>(datum.data().size(),datum.float_data_size());
		CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<size_in_datum;
		//	use pixel-char
		if (data.size() != 0) {
			CHECK_EQ(data.size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
			}
		}
		//	use numeric
		else {
			CHECK_EQ(datum.float_data_size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) 
				sum_blob.set_data(i, sum_blob.data(i) +(float)datum.float_data(i));
		}
		count++;
		if (count % 10000 == 0) LOG(INFO) << "Processed " << count << " files." << endl;
		cursor->Next();
	}
	if (count % 10000 != 0) LOG(INFO) << "Processed " << count << " files." << endl;
	for (int i = 0; i < sum_blob.data_size(); ++i) sum_blob.set_data(i, sum_blob.data(i) / count);
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		writeProtoToBinaryFile(sum_blob, argv[2]);
	}
	const int channels = sum_blob.shape().dim(1);
	const int dim = sum_blob.shape().dim(2)*sum_blob.shape().dim(3);
	vector<float> mean_values(channels, 0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) mean_values[c] += sum_blob.data(dim * c + i);
		LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
	}
	std::system("pause");
	return 0;
}
