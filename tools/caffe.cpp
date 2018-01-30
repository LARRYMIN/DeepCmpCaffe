#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>

#include "boost/algorithm/string.hpp"
#include "boost/shared_ptr.hpp"


#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/cmp_conv_layer.hpp"
#include "caffe/layers/cmp_inner_product_layer.hpp"


#define PLOG 0


using namespace std;
using namespace caffe;



using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(cm, "",
    "The compressed weights file, "
    "separated by ','. Cannot be set simultaneously with snapshot.");

DEFINE_int32(iterations, 20,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);



  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);






// Test: score a model.
int test1() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);





  const vector<shared_ptr<Layer<float> > >&  lays = caffe_net.layers();


 // Layer<float> l0 = lays[0];

cout<<"000000"<<endl;

   lays[0]->ComputeBlobMask();
  
cout<<"000000wwwwww"<<endl;
  vector<shared_ptr<Blob<float> > >& blob = lays[0]->blobs();

cout<<"11111"<<endl;

   //const float* weight ;
   const float* weight = (blob[0])->cpu_data();

cout<<"11111aaaaa"<<endl;
   
     int count = blob[0]->count();

cout<<"2222"<<endl;

     for (int i = 0; i < count; ++i)
  {
     //sort_weight[i] = fabs(weight[i]);
     cout<<weight[i]<<"  ";
  }





  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test1);








// compress
int compress() {

    

    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);

    caffe::NetParameter msg; 
    cout<<"FLAGS_weights   "<<FLAGS_weights<<endl;

    std::fstream input(FLAGS_weights.c_str(), ios::in | ios::binary); 
     if (!msg.ParseFromIstream(&input))       // function ParseFromIstream  is in  ::google::protobuf::Message
      { 
        cerr << "Failed to parse address book." << endl; 
       return -1; 
     } 
   //printf("length = %d\n", length);
   printf("Repeated Size = %d\n", msg.layer_size());
 
   ::google::protobuf::RepeatedPtrField< LayerParameter >* layer = msg.mutable_layer();


   cout<<msg.input_shape_size()<<endl ;
   cout<<"ff000000"<<endl;

   Net<float> caffe_net(FLAGS_model, caffe::TEST);
   caffe_net.CopyTrainedLayersFrom(msg);

   
    cout<<"FLAGS_cm   "<<FLAGS_cm<<endl;
    std::ofstream fout(FLAGS_cm.c_str(), std::ios::binary);
    int write_bytes=0;



   //Net<float> caffe_net();
   //caffe_net.CopyTrainedLayersFrom(msg);


   const vector<shared_ptr<Layer<float> > >&  lays = caffe_net.layers();
  

 // Layer<float> l0 = lays[0];

   //cout<<"000000"<<endl;
   cout<<"lays size  = "<<lays.size()<<endl;


   for(int i=0;i<lays.size();i++){

     cout<<endl<<"---------------------------------------------------"<<endl;

    vector<shared_ptr<Blob<float> > >& blob = lays[i]->blobs();

    //cout<<i<<"  "<<"blob size  = "<<blob.size()<<endl;


    if(blob.size()>0){
           cout << lays[i]->layer_param().name() << endl;
           cout << lays[i]->layer_param().type() << endl;

		   vector<shared_ptr<Blob<float> > >& blob = lays[i]->blobs();

           //Blob<int>& masks  = lays[i]->masks();
           const int *mask_data = lays[i]->masks().cpu_data();


		   //cout<<"11111"<<endl;

		   cout<<"blob size  = "<<blob.size()<<endl;


		   int count = blob[0]->count();

		   cout<<i<<"   "<<"count = "<<count<<endl;



		   //const float* weight ;
		   const float* weight = blob[0]->cpu_data();

           float* muweight = blob[0]->mutable_cpu_data();





		   const float* bias = blob[1]->cpu_data();

 		    	int countb = blob[1]->count();
			cout<<i<<"   "<<"countb = "<<countb<<endl;



		  // cout<<"11111aaaaa"<<endl;
           const float *cent_data = lays[i]->centroids().cpu_data();
           const int *indice_data = lays[i]->indices().cpu_data();


             


           if (strcmp(lays[i]->layer_param().type().c_str(), "CmpConvolution") == 0){

               const  shared_ptr<Layer<float> > l = lays[i];
               const  shared_ptr<BaseConvolutionLayer<float> > bcl=  boost::dynamic_pointer_cast<BaseConvolutionLayer<float> >(l) ;

               int class_num = bcl->class_num();

               cout<<"class_num  "<<class_num<<"  ";



	            for(int k=0;k<class_num;k++){
                   
			        //cout<<cent_data[k]<<"   ";
			        fout.write((char*)(cent_data+k), sizeof(float));
                    write_bytes+=sizeof(float);
		        }


           }

          if (strcmp(lays[i]->layer_param().type().c_str(), "CmpInnerProduct") == 0){

               const  shared_ptr<Layer<float> > l = lays[i];
               const  shared_ptr<CmpInnerProductLayer<float> > bcl=  boost::dynamic_pointer_cast<CmpInnerProductLayer<float> >(l) ;

               int class_num = bcl->class_num();

               cout<<"class_num  "<<class_num<<"  ";



	            for(int k=0;k<class_num;k++){
                   
			        //cout<<cent_data[k]<<"   ";
			        fout.write((char*)(cent_data+k), sizeof(float));
                    write_bytes+=sizeof(float);                
		        }


           }


          int count_one=0;
          for (int i = 0; i < count; ++i)
		  {

		        //sort_weight[i] = fabs(weight[i]);
		        //cout<<weight[i]<<"  ";
		        //cout<<mask_data[i]<<"  ";
		        //cout<<indice_data<<"  ";
                //unsigned char indice=(unsigned char)(indice_data[i]);
                //printf("%u  ",indice);

			    if(mask_data[i]){
                    count_one++;


			    }


		  }

          // write the number of mask = 1
      		fout.write((char*)(&count_one), sizeof(int));	
            write_bytes+=sizeof(int);	
 cout<<endl<<"--------------------------------count_one    "<<count_one<<"/"<<count<<"---------------------------------------"<<endl;
                   

		  for (int i = 0; i < count; ++i)
		  {
		        //sort_weight[i] = fabs(weight[i]);
		        //cout<<weight[i]<<"  ";
		        //cout<<mask_data[i]<<"  ";
		        //cout<<indice_data<<"  ";
                unsigned char indice=(unsigned char)(indice_data[i]);
                //printf("%u  ",indice);

			    if(mask_data[i]){
	          		fout.write((char*)(&i), sizeof(int));	
                    fout.write((char*)(&indice), sizeof(char));
                    write_bytes+=sizeof(int)+sizeof(char);	
                    if(i<100)
                    cout<<"["<<i<<",  "<<+indice<<"]   ";
			    }


		  }

	  	  for (int i = 0; i < countb; ++i)
		  {
		     //sort_weight[i] = fabs(weight[i]);
		     //cout<<weight[i]<<"  ";
			 //cout<<bias[i]<<"  ";
		     //cout<<mask_data[i]<<"  ";
		     //cout<<indice_data<<"  ";
             fout.write((char*)(bias+i), sizeof(float));	
             write_bytes+=sizeof(float);  

		  }


          for (int j = 0; j < count; ++j)
		  {
		     //cout<<indice_data[j]<<"  ";
             //printf("%cu  ",indice_data[j]);
             //cout<<"["<<weight[j]<<",  "<<muweight[j]<<"]   ";
             if(j==100)break;

		  }

    }


   }
  
  
    fout.close();

    cout<<"-----------------writed bytes   "<<write_bytes<<"----------------------------"<<endl;


   //CopyTrainedLayersFrom(msg);


   ::google::protobuf::RepeatedPtrField< LayerParameter >::iterator it = layer->begin();
   for (; it != layer->end(); ++it)
   {
     cout << it->name() << endl;
     cout << it->type() << endl;
     cout << it->convolution_param().weight_filler().max() << endl;
     //cout << it->convolution_param().weight_filler().value() << endl;

/*

     cout<<"55555000000wwwwww"<<endl;
     vector<shared_ptr<Blob<float> > >& blob = (*it)->blobs();

      int count = blob[0]->count();

     cout<<"count2 = "<<count<<endl;


     const float* weight = blob[0]->cpu_data();
     


     for (int i = 0; i < count; ++i)
     {
      //sort_weight[i] = fabs(weight[i]);
      cout<<weight[i]<<"  ";
     }

*/
    cout <<"layer.blobs_size()  "<< it->blobs_size()<<endl;


  //sparse parameters
  float sparse_ratio_;
  int class_num_;
  bool quantize_term_;

    sparse_ratio_ = it->convolution_param().sparse_ratio();
    class_num_ = it->convolution_param().class_num();
    quantize_term_ = it->convolution_param().quantize_term();


    cout<<"sparse_ratio_ "<<sparse_ratio_ <<endl;
    cout<<"class_num_ "<< class_num_ <<endl;
    cout<<"quantize_term_ "<<quantize_term_ <<endl;

/*

	 int count = ((CmpConvolutionLayer<float>)it)->blobs_[0]->count();

	 const Dtype* weight = it->blobs_[0]->cpu_data();
	  //Dtype min_weight = weight[0] , max_weight = weight[0];
	  vector<Dtype> sort_weight(count);
		       
	  for (int i = 0; i < count; ++i)
	  {
	    //this->masks_[i] = 1; //initialize
	     sort_weight[i] = fabs(weight[i]);
	     cout<<weight[i]<<"  ";
	  }

	  cout<<endl;


*/


   } 

  return 0;


}


RegisterBrewFunction(compress);







//de compress
int decmp() {

    

    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);

    caffe::NetParameter msg; 


/*
     cout<<"FLAGS_weights   "<<FLAGS_weights<<endl;

     std::fstream input(FLAGS_weights.c_str(), ios::in | ios::binary); 
     if (!msg.ParseFromIstream(&input))       // function ParseFromIstream  is in  ::google::protobuf::Message
      { 
        cerr << "Failed to parse address book." << endl; 
       return -1; 
     } 
     //printf("length = %d\n", length);
     printf("Repeated Size = %d\n", msg.layer_size());
 
    ::google::protobuf::RepeatedPtrField< LayerParameter >* layer = msg.mutable_layer();


     cout<<msg.input_shape_size()<<endl ;
*/



     Net<float> caffe_net(FLAGS_model, caffe::TEST);
     //caffe_net.CopyTrainedLayersFrom(msg);
     int read_bytes=0;
   
    
     //std::ofstream fout("examples/mnist/compress/lenet.cm", std::ios::binary);

     cout<<"FLAGS_cm   "<<FLAGS_cm<<endl;
     std::ifstream fin(FLAGS_cm.c_str(), std::ios::binary);
     const vector<shared_ptr<Layer<float> > >&  lays = caffe_net.layers();


     cout<<"lays size  = "<<lays.size()<<endl;


     for(int i=0;i<lays.size();i++){

        cout<<endl<<"---------------------------------------------------"<<endl;

        vector<shared_ptr<Blob<float> > >& blob = lays[i]->blobs();

        cout<<i<<"  "<<"blob size  = "<<blob.size()<<endl;


        if(blob.size()>0){


		       vector<shared_ptr<Blob<float> > >& blob = lays[i]->blobs();

               //Blob<int>& masks  = lays[i]->masks();
               int *mask_data = lays[i]->masks().mutable_cpu_data();


               int count = blob[0]->count();

		       cout<<i<<"   "<<"count = "<<count<<endl;



		       //const float* weight ;
		       float* weight = blob[0]->mutable_cpu_data();
		       float* bias = blob[1]->mutable_cpu_data();

 		       int countb = blob[1]->count();
			   cout<<i<<"   "<<"countb = "<<countb<<endl;
                  float *cent_data = lays[i]->centroids().mutable_cpu_data();
                  int *indice_data = lays[i]->indices().mutable_cpu_data();


                   cout << lays[i]->layer_param().name() << endl;
                   cout << lays[i]->layer_param().type() << endl;  


                   if (strcmp(lays[i]->layer_param().type().c_str(), "CmpConvolution") == 0){

		               const  shared_ptr<Layer<float> > l = lays[i];
                       const  shared_ptr<BaseConvolutionLayer<float> > bcl=  boost::dynamic_pointer_cast<BaseConvolutionLayer<float> >(l) ;

                       int class_num = bcl->class_num();

                       cout<<"class_num  "<<class_num<<"  ";



			            for(int k=0;k<class_num;k++){
                           
					        
					        fin.read((char*)(cent_data+k), sizeof(float));
                            read_bytes+=sizeof(float);
                            //cout<<cent_data[k]<<"   ";
				        }
   
 
                   }

                  if (strcmp(lays[i]->layer_param().type().c_str(), "CmpInnerProduct") == 0){

		               const  shared_ptr<Layer<float> > l = lays[i];
                       const  shared_ptr<CmpInnerProductLayer<float> > bcl=  boost::dynamic_pointer_cast<CmpInnerProductLayer<float> >(l) ;

                       int class_num = bcl->class_num();

                       cout<<"class_num  "<<class_num<<"  ";



			            for(int k=0;k<class_num;k++){
                           
					        
					        fin.read((char*)(cent_data+k), sizeof(float));
                            //cout<<cent_data[k]<<"   ";
                            read_bytes+=sizeof(float);                
				        }
   
 
                   }



            int count_one=0;
            fin.read((char*)(&count_one), sizeof(int));	
            read_bytes+=sizeof(int);

          cout<<endl<<"--------------------------------count_one    "<<count_one<<"---------------------------------------"<<endl;


          for (int i = 0; i < count; ++i)
		  {

		      mask_data[i]=0;
              weight[i]=0;

		  }
                   

		  for (int i = 0; i < count_one; ++i)
		  {

		        //sort_weight[i] = fabs(weight[i]);
		        //cout<<weight[i]<<"  ";
		        //cout<<mask_data[i]<<"  ";
		        //cout<<indice_data[i]<<"  ";
                //unsigned char indice=(unsigned char)(indice_data[i]);
                //printf("%u  ",indice);

                int index;
                unsigned char indice;

                fin.read((char*)(&index), sizeof(int));
                fin.read((char*)(&indice), sizeof(char));
                read_bytes+=sizeof(int)+sizeof(char);	

                mask_data[index] =1;
                indice_data[index] = (int)indice; 


                 if(i<100)
                    cout<<"["<<index<<",  "<<+indice<<"]   ";

                //cout<<"-----index  "<<index<<endl; 
                //printf("-----indice    %u\n", indice);
                //cout<<"-----cent_data[indice]  "<<cent_data[indice]<<endl; 

                weight[index] = cent_data[indice];

		  }



          for (int i = 0; i < count; ++i)
		  {

		        //sort_weight[i] = fabs(weight[i]);
		        //cout<<weight[i]<<"  ";
		        //cout<<mask_data[i]<<"  ";
		        //cout<<indice_data<<"  ";
                //unsigned char indice=(unsigned char)(indice_data[i]);
                //printf("%u  ",indice);

		  }


    






	  	  for (int i = 0; i < countb; ++i)
		  {
		     //sort_weight[i] = fabs(weight[i]);
		     //cout<<weight[i]<<"  ";
			 //cout<<bias[i]<<"  ";
		     //cout<<mask_data[i]<<"  ";
		     //cout<<indice_data<<"  ";
              fin.read((char*)(bias+i), sizeof(float));	
              read_bytes+=sizeof(float);  

		  }
  
          const float* weight1 = blob[0]->cpu_data();

          for (int j = 0; j < count; ++j)
		  {
            
		    //cout<<indice_data[j]<<"  ";
            //printf("%cu  ",indice_data[j]);

            //cout<<"["<<weight1[j]<<",  "<<weight[j]<<"]   ";

            if(j==100)break;

		  }




         }




     }


    fin.close();

    cout<<endl<<"-----------------read_bytes   "<<read_bytes<<"----------------------------"<<endl;


  


  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        cout<<"jk=  "<<j<<"   "<<k<<"   "<<score<<endl;
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  cout<<endl<<"test_score.size()  "<<test_score.size()<<endl;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }






}
RegisterBrewFunction(decmp);





































// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
