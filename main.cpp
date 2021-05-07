#include "mbed.h"
#include "mbed_rpc.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "uLCD_4DGL.h"
#include "stm32l475e_iot01_accelero.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "mbed_events.h"
#include "math.h"
using namespace std::chrono;

WiFiInterface *wifi;
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char* topic = "Mbed";
int16_t accdata[3] = {0};
uLCD_4DGL uLCD(D1, D0, D2);// serial tx, serial rx, reset pin;
DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
DigitalOut myled3(LED3);
DigitalIn btn_confirm(USER_BUTTON);
BufferedSerial pc(USBTX, USBRX);
void Gesture_UI(Arguments *in, Reply *out);
void Tilt_Detection(Arguments *in, Reply *out);
RPCFunction gesture(&Gesture_UI, "Gesture_UI");
RPCFunction tilt(&Tilt_Detection, "Tilt_Detection");

double x, y;
int flag1 = 0, flag2 = 0, mode = 0, Threshold_Angle = 0, receive_angle = 0, tilt_mode = 0, tilt_cnt = 0, init1 = 0, init2 = 1; 

Thread t1,t2,t3;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

//Thread mqtt_thread(osPriorityHigh);
//EventQueue mqtt_queue;
EventQueue q1, q2, q3;

void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
//if (receive_angle) {
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
//    printf(msg);
//    ThisThread::sleep_for(10ms);
    char payload[300];

if (receive_angle) {
    sprintf(payload, "Selected Theshold Angle is :%.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);	
	char buf[100] = "/Gesture_UI/run 0";
	char outbuf[256];
	RPC::call(buf, outbuf);
	receive_angle = 0;
}
if (tilt_mode) {
    sprintf(payload, "Over Threshold Angle ! :%.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
	tilt_cnt++;
	if (tilt_cnt >= 10) {
		tilt_cnt = 0;
		printf("Over Threshold Angle more than 10 times ! Back to RPC Loop.\n");
		char buf[100] = "/Tilt_Detection/run 0";
		char outbuf[256];
		RPC::call(buf, outbuf);
		uLCD.cls();
	}
	tilt_mode = 0;
}
//	printf("%s\r\n", outbuf);
    ++arrivedcount;
}

void close_mqtt() {
    closed = true;
}


// Return the result of the last prediction
int PredictGesture(float* output) {
	// How many times the most recent gesture has been matched in a row
	static int continuous_count = 0;
	// The result of the last prediction
	static int last_predict = -1;

	// Find whichever output has a probability > 0.8 (they sum to 1)
	int this_predict = -1;
	for (int i = 0; i < label_num; i++) {
		if (output[i] > 0.8) this_predict = i;
	}

	// No gesture was detected above the threshold
	if (this_predict == -1) {
		continuous_count = 0;
		last_predict = label_num;
		return label_num;
	}

	if (last_predict == this_predict) {
		continuous_count += 1;
	} else {
		continuous_count = 0;
	}
	last_predict = this_predict;

	// If we haven't yet had enough consecutive matches for this gesture,
	// report a negative result
	if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
		return label_num;
	}
	// Otherwise, we've seen a positive result, so clear all our variables
	// and report it
	continuous_count = 0;
	last_predict = -1;

	return this_predict;
}


int gesture_main(MQTT::Client<MQTTNetwork, Countdown>* client) {

	// Whether we should clear the buffer next time we fetch data
	bool should_clear_buffer = false;
	bool got_data = false;



	// The gesture index of the prediction
	int gesture_index;

	// Set up logging.
	static tflite::MicroErrorReporter micro_error_reporter;
	tflite::ErrorReporter* error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
				"Model provided is schema version %d not equal "
				"to supported version %d.",
				model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}

	// Pull in only the operation implementations we need.
	// This relies on a complete list of all the ops needed by this graph.
	// An easier approach is to just use the AllOpsResolver, but this will
	// incur some penalty in code space for op implementations that are not
	// needed by this graph.
	static tflite::MicroOpResolver<6> micro_op_resolver;
	micro_op_resolver.AddBuiltin(
			tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
			tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
	micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
															 tflite::ops::micro::Register_MAX_POOL_2D());
	micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
															 tflite::ops::micro::Register_CONV_2D());
	micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
															 tflite::ops::micro::Register_FULLY_CONNECTED());
	micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
															 tflite::ops::micro::Register_SOFTMAX());
	micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
															 tflite::ops::micro::Register_RESHAPE(), 1);

	// Build an interpreter to run the model with
	static tflite::MicroInterpreter static_interpreter(
			model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
	tflite::MicroInterpreter* interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors
	interpreter->AllocateTensors();

	// Obtain pointer to the model's input tensor
	TfLiteTensor* model_input = interpreter->input(0);
	if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
			(model_input->dims->data[1] != config.seq_length) ||
			(model_input->dims->data[2] != kChannelNumber) ||
			(model_input->type != kTfLiteFloat32)) {
		error_reporter->Report("Bad input tensor parameters in model");
		return -1;
	}

	int input_length = model_input->bytes / sizeof(float);

	TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
	if (setup_status != kTfLiteOk) {
		error_reporter->Report("Set up failed\n");
		return -1;
	}

	error_reporter->Report("Set up successful...\n");



	MQTT::Message message;
	char buff[100];

	while (true) {

		if (flag1) myled1 = 1;
		else
			myled1 = 0;

		if (flag1 && !init1) {
			printf("current_mode : 25\n");
			Threshold_Angle = 25;
			mode = 0;
			// setting of uLCD
			uLCD.cls();
			uLCD.locate(0,0);
			uLCD.color(GREEN);
			uLCD.printf("Select angle\n"); //Default Green on black text



			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,1);
			uLCD.printf("25");    
/*
			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,3);
			uLCD.printf("30");  

			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,5);
			uLCD.printf("35");  

			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,7);
			uLCD.printf("40");
*/  
			init1 = 1;

		}

		// Attempt to read new data from the accelerometer
		got_data = ReadAccelerometer(error_reporter, model_input->data.f,
																 input_length, should_clear_buffer);

		// If there was no new data,
		// don't try to clear the buffer again and wait until next time
		if (!got_data) {
			should_clear_buffer = false;
			continue;
		}

		// Run inference, and report any error
		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			error_reporter->Report("Invoke failed on index: %d\n", begin_index);
			continue;
		}

		TfLiteTensor* output = interpreter->output(0);
		float value = output->data.f[0];

		// Analyze the results to obtain a prediction
		gesture_index = PredictGesture(interpreter->output(0)->data.f);

		// Clear the buffer next time we read data
		should_clear_buffer = gesture_index < label_num;

		

		// Produce an output
		if ((gesture_index < label_num) && flag1 && (gesture_index == 0) && mode < 3)
			mode += 1;
		if ((gesture_index < label_num) && flag1 && (gesture_index == 2) && mode > 0)
			mode -= 1;
		switch (mode) {
			case 0:
				Threshold_Angle = 25;
				break;
			case 1:
				Threshold_Angle = 30;
				break;
			case 2:
				Threshold_Angle = 35;
				break;
			case 3:
				Threshold_Angle = 40;
				break;
			default :
				Threshold_Angle = 25;
				break;
		}	

		if ((gesture_index < label_num) && flag1 && init1) {
//			printf("predicr_label : %d\n",gesture_index);
			printf("current_mode : %d\n", Threshold_Angle);
			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,1);
			uLCD.printf("%d", Threshold_Angle); 			
			error_reporter->Report(config.output_message[gesture_index]);
		}
		if (!btn_confirm && flag1) {
			message_num++;

			sprintf(buff, "%d", Threshold_Angle);
//			printf("%d\n",Threshold_Angle);
    		message.qos = MQTT::QOS0;
    		message.retained = false;
    		message.dup = false;
    		message.payload = (void*) buff;
    		message.payloadlen = strlen(buff) + 1;
    		int rc = client->publish(topic, message);
			receive_angle = 1;
			ThisThread::sleep_for(500ms);
//    		printf("rc:  %d\r\n", rc);
//    		printf("Puslish message: %s\r\n", buff);			
//			printf("%d\n", receive_angle);
		}


/*
		if (mode == 0) {
				uLCD.line(20,12,102,12,RED);
				uLCD.line(20,12,20,32,RED);
				uLCD.line(102,12,102,32,RED);
				uLCD.line(20,32,102,32,RED);
				uLCD.line(20,45,102,45,BLACK);
				uLCD.line(20,45,20,65,BLACK);
				uLCD.line(102,45,102,65,BLACK);
				uLCD.line(20,65,102,65,BLACK); 
				uLCD.line(20,77,102,77,BLACK);
				uLCD.line(20,77,20,97,BLACK);
				uLCD.line(102,77,102,97,BLACK);
				uLCD.line(20,97,102,97,BLACK);
				uLCD.line(20,108,102,108,BLACK);
				uLCD.line(20,108,20,127,BLACK);
				uLCD.line(102,108,102,127,BLACK);                         
		}
		if (mode == 1) {
				uLCD.line(20,12,102,12,BLACK);
				uLCD.line(20,12,20,32,BLACK);
				uLCD.line(102,12,102,32,BLACK);
				uLCD.line(20,32,102,32,BLACK);
				uLCD.line(20,45,102,45,RED);
				uLCD.line(20,45,20,65,RED);
				uLCD.line(102,45,102,65,RED);
				uLCD.line(20,65,102,65,RED); 
				uLCD.line(20,77,102,77,BLACK);
				uLCD.line(20,77,20,97,BLACK);
				uLCD.line(102,77,102,97,BLACK);
				uLCD.line(20,97,102,97,BLACK);
				uLCD.line(20,108,102,108,BLACK);
				uLCD.line(20,108,20,127,BLACK);
				uLCD.line(102,108,102,127,BLACK);          
		}
		if (mode == 2) {
				uLCD.line(20,12,102,12,BLACK);
				uLCD.line(20,12,20,32,BLACK);
				uLCD.line(102,12,102,32,BLACK);
				uLCD.line(20,32,102,32,BLACK);
				uLCD.line(20,45,102,45,BLACK);
				uLCD.line(20,45,20,65,BLACK);
				uLCD.line(102,45,102,65,BLACK);
				uLCD.line(20,65,102,65,BLACK); 
				uLCD.line(20,77,102,77,RED);
				uLCD.line(20,77,20,97,RED);
				uLCD.line(102,77,102,97,RED);
				uLCD.line(20,97,102,97,RED);
				uLCD.line(20,108,102,108,BLACK);
				uLCD.line(20,108,20,127,BLACK);
				uLCD.line(102,108,102,127,BLACK);              
		}
		if (mode == 3) {
				uLCD.line(20,12,102,12,BLACK);
				uLCD.line(20,12,20,32,BLACK);
				uLCD.line(102,12,102,32,BLACK);
				uLCD.line(20,32,102,32,BLACK);
				uLCD.line(20,45,102,45,BLACK);
				uLCD.line(20,45,20,65,BLACK);
				uLCD.line(102,45,102,65,BLACK);
				uLCD.line(20,65,102,65,BLACK); 
				uLCD.line(20,77,102,77,BLACK);
				uLCD.line(20,77,20,97,BLACK);
				uLCD.line(102,77,102,97,BLACK);
				uLCD.line(20,97,102,97,BLACK);
				uLCD.line(20,108,102,108,RED);
				uLCD.line(20,108,20,127,RED);
				uLCD.line(102,108,102,127,RED);             
		}
*/
//		ThisThread::sleep_for(10ms);

	}

}

void client_yield(MQTT::Client<MQTTNetwork, Countdown> *client)
{
    while (1) {
        if (closed) break;
        client->yield(500);
        ThisThread::sleep_for(500ms);
    }
}

int tilt_main(MQTT::Client<MQTTNetwork, Countdown> *client)
{
	int16_t acc_current[3] = {0};
//	int stdn[3] = {0,0,1000};
	int dot = 0;
	float la = 0.0, lb = 0.0, result;
	MQTT::Message message;
	char buff[100];

	while (1) {

		if (flag2) {
			myled2 = 1;
			if (!init2)
				myled3 = 1;
			else
				myled3 = 0;
		}
		else {
			myled2 = 0;
			myled3 = 0;
		}
		ThisThread::sleep_for(100ms);

		BSP_ACCELERO_AccGetXYZ(accdata);
		if (!btn_confirm && flag2 && !init2) {
			for (int i = 0; i < 3; i++)
				acc_current[i] = accdata[i];
			printf("Initialization Completed !\n");
			myled3 = 0;
			init2 = 1;
			uLCD.cls();
			uLCD.locate(0,0);
			uLCD.color(RED);
			uLCD.printf("Tilt Angle : \n");	
			ThisThread::sleep_for(100ms);
		}

		la = 0.0; lb = 0.0; dot = 0;
		for (int i = 0; i < 3; i++) {
			dot += acc_current[i] * accdata[i];
			la += 1.0 * (accdata[i] * accdata[i]);
			lb += 1.0 * (acc_current[i] * acc_current[i]);
		}
	
		la = sqrt(la);
		lb = sqrt(lb);
/*
		if (accdata[2] > acc_current[2])
			result = acos(acc_current[2]*1.0 / accdata[2]) * 180 / 3.1415926;
		else
			result = acos(accdata[2]*1.0 / acc_current[2]) * 180 / 3.1415926;
*/
		result = acos(dot * 1.0 / (la * lb)) * 180 / 3.1415926;
//		ThisThread::sleep_for(100ms);

		if (flag2 && init2) {
			uLCD.text_width(2);
			uLCD.text_height(2);
			uLCD.color(BLUE);
			uLCD.locate(2,1);
			uLCD.printf("%.2f",result);
		}    

		if (result > float(Threshold_Angle) && flag2 && init2) {
			message_num++;

			sprintf(buff, "%.2f", result);
//			printf("%.2f\n",result);
    		message.qos = MQTT::QOS0;
    		message.retained = false;
    		message.dup = false;
    		message.payload = (void*) buff;
    		message.payloadlen = strlen(buff) + 1;
    		int rc = client->publish(topic, message);
//    		printf("rc:  %d\r\n", rc);
//    		printf("Puslish message: %s\r\n", buff);
			tilt_mode = 1;
//			ThisThread::sleep_for(200ms);			
		}
		ThisThread::sleep_for(500ms);
	}
	





	while (1) {
		ThisThread::sleep_for(100ms);
	}





}




int main(void) {
	//The mbed RPC classes are now wrapped to create an RPC enabled version - see RpcClasses.h so don't add to base class

	BSP_ACCELERO_Init();
	// MQTT Setting
    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
            printf("ERROR: No WiFiInterface found.\r\n");
            return -1;
    }


    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            return -1;
    }


	NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
	MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

    //TODO: revise host to your IP
    const char* host = "172.20.10.2";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
            printf("Connection error.");
            return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
            printf("Fail to subscribe\r\n");
    }

	// Start two thread
	t1.start(callback(&q1, &EventQueue::dispatch_forever));
	q1.call(&gesture_main, &client);
	t3.start(callback(&q3, &EventQueue::dispatch_forever));
	q3.call(&client_yield, &client);
	t2.start(callback(&q2, &EventQueue::dispatch_forever));
	q2.call(&tilt_main, &client);	
//    mqtt_thread.start(callback(queue, &EventQueue::dispatch_forever));
	// receive commands, and send back the responses
	char buf[256], outbuf[256];



	FILE *devin = fdopen(&pc, "r");
	FILE *devout = fdopen(&pc, "w");
 
	while(1) {
		memset(buf, 0, 256);
		for (int i = 0; ; i++) {
			char recv = fgetc(devin);
			if (recv == '\n') {
				printf("\r\n");
				break;
			}
			buf[i] = fputc(recv, devout);
		}
		//Call the static call method on the RPC class
		RPC::call(buf, outbuf);
		printf("%s\r\n", outbuf);	
	}
}

// Make sure the method takes in Arguments and Reply objects.
void Gesture_UI (Arguments *in, Reply *out)   {
	bool success = true;

		// In this scenario, when using RPC delimit the two arguments with a space.
	x = in->getArg<double>();
//    y = in->getArg<double>();

		// Have code here to call another RPC function to wake up specific led or close it.
//    char buffer[200], outbuf[256];
//    char strings[20];
	flag1 = x;
	init1 = 0;
//    int on = y;

}

void Tilt_Detection (Arguments *in, Reply *out)   {
	bool success = true;

		// In this scenario, when using RPC delimit the two arguments with a space.
	y = in->getArg<double>();
//    y = in->getArg<double>();

		// Have code here to call another RPC function to wake up specific led or close it.
//    char buffer[200], outbuf[256];
//    char strings[20];
	flag2 = y;
	init2 = 0;
//    int on = y;

}
