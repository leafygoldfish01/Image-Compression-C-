#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "pixel.h"

using namespace cv;
using namespace std;

vector<Pixel> clusterCentres;
Mat image;
int numberofcolors;
Mat labels;


static double euclideanDistance(int x1, int y1, int c1, int x2, int y2, int c2)
{
    int tem1=pow(x1 -x2,2);
    int tem2=pow(y1 -y2,2);
    int tem3=pow(c1 -c2,2);
    return sqrt(tem1+tem2+tem3);
}

static int random(int lim)
{
    uniform_int_distribution<int> uid{0, lim}; // uid is a uniform random no generator
    default_random_engine dre(chrono::steady_clock::now().time_since_epoch().count());// dre is a random no generator
    int ret = uid(dre); // returns a random no between 0 and lim
    return ret;
}
void newClusterCenters()
{
    int rw=image.rows;
    int cl=image.cols;
    for(int i=0;i<rw;i++){
        for(int j=0;j<cl;j++){
            Vec3b img_pixel = image.at<Vec3b>(i,j);
            int centroidLabel = 0;
            int b1=img_pixel[0];
            int g1=img_pixel[1];
            int r1=img_pixel[2];
            double mindistance=DBL_MAX;
            for(int t=0;t<numberofcolors;t++){
                int val1=clusterCentres[t].b;
                int val2=clusterCentres[t].g;
                int val3=clusterCentres[t].r;
                double distance=euclideanDistance(val1,val2,val3,b1,g1,r1);
                if(distance<mindistance){
                    mindistance=distance;
                    centroidLabel=t;
                    labels.at<uchar>(i,j)=(uchar)centroidLabel;
                }
            }
        }
    }
    
}
void findCentroids(){
    int rw=image.rows;
    int cl=image.cols;
    
    for(int i=0;i<numberofcolors;i++){
        double val1=0;
        double val2=0;
        double val3=0;
        int it=0;
        for(int j=0;j<rw;j++){
            for(int k=0;k<cl;k++){
                int temp = labels.at<uchar>(j,k);
                if(temp==i){
                    Vec3b img_pixel = image.at<Vec3b>(j,k);
                    val1+=img_pixel[0];
                    val2+=img_pixel[1];
                    val3+=img_pixel[2];
                    it++;
                }
            }
        }
        clusterCentres.at(i) = Pixel(val1/it,val2/it,val3/it);
    }
}


void train_model(int it){
    cout<<"Image Compression Started"<<endl;
    newClusterCenters();
    for(int i=0;i<it;i++){
        findCentroids();
        newClusterCenters();
        cout<<"Working on Compression Step: "<<i+1<<endl;
    }
}

int main(int argc, char **argv)
{
    
    string inputImage = "./image.png";
    numberofcolors = 20;
    cout<<"Enter the input image address: ";
    cin>>inputImage;
    cout<<"Enter the Number of colors you want to take in image: ";
    cin>>numberofcolors;
    string outputImage ="compressed_image.png";
    int steps=61;
    while(steps<20 || steps==61){
        cout<<"Enter the number of steps you want to train the model: ";
        cin>>steps;
        if(steps==61){
            break;
        }
        else if(steps<10){
            cout<<"Note: Steps Entered must be higher for better quality of image"<<endl;
            cout<<"Re-enter the number of steps you want to train the model: ";
        }
    }
    Mat image1 = imread(inputImage);

    //jpeg compression for comparison
    vector<uchar> buffer;
    imencode(".jpg", image1, buffer,{IMWRITE_JPEG_QUALITY, 60});
    imwrite("jpeg_compressed.jpg", image1);

    //our model compression
    image = image1;
    if (image1.empty()){
        cout<<"incorrect image address"<<endl;
        return -1;
    }
    else{
        cout<<"image uploaded successfully"<<endl;
    }
    int img_rows = image.rows;
    int img_cols = image.cols;
    labels = Mat::zeros(Size(img_cols, img_rows), CV_8UC1);

    for (int i = 0; i < numberofcolors; i++){
        int randomR = random(img_rows-1);
        int randomC = random(img_cols-1);
        Vec3b img_pixel = image.at<Vec3b>(randomR, randomC);
        clusterCentres.push_back(Pixel(img_pixel[0], img_pixel[1], img_pixel[2]));
    }

    train_model(steps);

    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            Pixel img_pixel = clusterCentres.at(labels.at<uchar>(i, j));
            Vec3b cvt_pixel(img_pixel.b, img_pixel.g, img_pixel.r);
            image.at<Vec3b>(i, j) = cvt_pixel;
        }
    }

    //writing our compressed image
    imwrite(outputImage, image);

    //displaying our compressed image
    imshow("Display Our Model Image", image); 
    waitKey(0); 
    return 0;
}