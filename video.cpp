#include <sstream>
#include <string>
#include <iostream>
//#include <opencv2\highgui.h>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\cv.h>
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define COMMAND_FORWARD 'f'
#define COMMAND_BACKWARD 'b'
#define COMMAND_LEFT 'l'
#define COMMAND_RIGHT 'r'
#define COMMAND_STOP 's'

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

int B_H_MIN = 92;
int B_H_MAX = 255;
int B_S_MIN = 135;
int B_S_MAX = 255;
int B_V_MIN = 176;
int B_V_MAX = 255;

int G_H_MIN = 54;
int G_H_MAX = 255;
int G_S_MIN = 152;
int G_S_MAX = 255;
int G_V_MIN =  146;
int G_V_MAX = 227;


//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "Thresholded Image 2";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";

void error(const char *msg)
{
    perror(msg);
    exit(0);
}


void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//cout << "x,y: " << x << ", " << y;

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				//cout << x << "," << y;
				drawObject(x, y, cameraFeed);

			}


		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

void sendCommand(int socketfd, char command, char last_command)
{
	int n;
	char buffer[256];

	bzero(buffer,256);
    	strcpy(buffer, &command);
    	n = write(socketfd, buffer, strlen(buffer));
    	if (n < 0) 
         	error("ERROR writing to socket");
    	bzero(buffer,256);
    	n = read(socketfd, buffer, 255);
    	if (n < 0) 
         	error("ERROR reading from socket");
    	printf("Received: %s\n", buffer);
		last_command = command;
}

void onPositionUpdate(Point green, Point blue, Point front, int socketfd, char last_command)
{
	switch(robot)
	{
		//The robot is the green one
		case 0:
		{
			if(blue.X == 0 && blue.Y == 0)
			{
				sendCommand(socketfd, COMMAND_STOP, last_command);
				//win
			}
			else if(almostColinear(green, front, blue) && (last_command == COMMAND_LEFT || last_command == COMMAND_RIGHT))
			{
				sendCommand(socketfd, COMMAND_STOP, last_command);
				//backwards
				if(green.X - blue.X < front.X - blue.X)
				{
					sendCommand(socketfd, COMMAND_BACKWARD, last_command);
				}
				else
				{
					sendCommand(socketfd, COMMAND_FORWARD, last_command);
				}
			}
			else if(isLeft(green, front, blue)) {
				if(last_command != COMMAND_LEFT)
				{
					sendCommand(socketfd, COMMAND_STOP, last_command);
				}
				sendCommand(socketfd, COMMAND_LEFT, last_command);
			}
			else
			{
				if(last_command != COMMAND_RIGHT)
				{
					sendCommand(socketfd, COMMAND_STOP, last_command);
				}
				sendCommand(socketfd, COMMAND_RIGHT, last_command);
			}
			break;
		}
		//The robot is the blue one
		case 1:
		{
			break;
		}
	}
}

public bool isLeft(Point a, Point b, Point c){
     return ((b.X - a.X)*(c.Y - a.Y) - (b.Y - a.Y)*(c.X - a.X)) > 0;
}

public bool almostColinear(Point a, Point b, Point c){
     return ((b.X - a.X)*(c.Y - a.Y) - (b.Y - a.Y)*(c.X - a.X)) < 1 && ((b.X - a.X)*(c.Y - a.Y) - (b.Y - a.Y)*(c.X - a.X)) > -1;
}

int main(int argc, char* argv[])
{

	//some boolean variables for different functionality within this
	//program
	bool trackObjects = true;
	bool useMorphOps = true;

	int last_command = -1;
	
	//0 = GREEN; 1 = BLUE
	int robot = atoi(argv[1]);

	Point p;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	Mat threshold2;
	Mat threshold3;
	//x and y values for the location of the object
	int x_b = 0, y_b = 0;
	int x_g = 0, y_g = 0;
	int x_front = 0; y_front = 0;
	//create slider bars for HSV filtering
	createTrackbars();
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.open("rtmp://172.16.254.63/live/live");
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop


	int socketfd;
	int portno = 20232;
	struct sockaddr_in serv_addr;
    	struct hostent *server;

	socketfd = socket(AF_INET, SOCK_STREAM, 0);
    	if (socketfd < 0) 
        	error("ERROR opening socket");

	string dest = "193.226.12.217";

	//server = gethostbyaddr(dest.c_str(), dest.length(), AF_INET);
	server = gethostbyname("193.226.12.217");
        if (server == NULL) {
           	fprintf(stderr,"ERROR, no such host\n");
        	exit(0);
        }

	bzero((char *) &serv_addr, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr, 
         	(char *)&serv_addr.sin_addr.s_addr,
         	server->h_length);
        serv_addr.sin_port = htons(portno);


	if (connect(socketfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
        	error("ERROR connecting");

	//Test
	
	//sendCommand(socketfd, 'r');
	//sendCommand(socketfd, 's');
	
	
	while (1) {


		//store image to matrix
		capture.read(cameraFeed);
		//convert frame from BGR to HSV colorspace
		if(cameraFeed.empty())
		{
		    	continue;
		}	
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matrix

		inRange(HSV, Scalar(B_H_MIN, B_S_MIN, B_V_MIN), Scalar(B_H_MAX, B_S_MAX, B_V_MAX), threshold);

		//Test detect object
		//inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

		inRange(HSV, Scalar(G_H_MIN, G_S_MIN, G_V_MIN), Scalar(G_H_MAX, G_S_MAX, G_V_MAX), threshold2);
		
		inRange(HSV, Scalar(Y_H_MIN, Y_S_MIN, Y_V_MIN), Scalar(Y_H_MAX, Y_S_MAX, Y_V_MAX), threshold3);
		
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps)
		{
			morphOps(threshold);
			morphOps(threshold2);
		}
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if (trackObjects)
		{
			trackFilteredObject(x_b, y_b, threshold, cameraFeed);
			trackFilteredObject(x_g, y_g, threshold2, cameraFeed);
			trackFilteredObject(x_front, y_front, threshold3, cameraFeed);
			
			Point green(x_g, y_g);
			Point blue(x_b, y_b);
			Point nose(x_front, y_front);
			/*
			if(robot == 0)
			{
				determineNosePosition(x_g, y_g, x_front, y_front,
			}*/
			onPositionUpdate(robot, green, blue, nose, socketfd, last_command);
		}

		//show frames
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);
		imshow(windowName1, threshold2);
		setMouseCallback("Original Image", on_mouse, &p);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(30);
	}
	

	close(socketfd);

	return 0;
}
