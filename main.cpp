//*****************************************************************************
//
//                                TEMPLATE FEDHA Code.
//                             Written  by Bilal Dastagir.
//                                March , 2nd, 2022
//
//******************************************************************************
// Your First OPENCV C++ Program

#include <stdio.h>
#include <iostream>

int main2(int argc, const char * argv[]) {
    std::cout << "Hello, World!\n";
    return 0;
}

#include "../opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
int main()
{
    std::string image_path = samples::findFile("starry_night.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
}
