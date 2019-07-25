#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>

using namespace cv;
using namespace std;

/*
convolution function(input image,
                     image.channels(),
                     image.rows,
                     image.cols,
                     filter size,
                     padding size,
                     stride size,
                     image.rows+2*padding,
                     image.cols+2*padding,
                     (image.rows-filter+2*padding)/stride + 1),
                     (image.rows-filter+2*padding)/stride + 1)
                     )
*/

double ***convolution(double ***image, int image_channel, int image_rows, int image_cols, int filter, int padding, int stride, int matrix_rows_size, int matrix_cols_size, int convolution_rows_size, int convolution_cols_size)
{
    double ***Weight_input;

    Weight_input=(double ***)malloc(image_channel*sizeof(double **));
    
    for(int i=0;i<image_channel;i++)
    {
        *(Weight_input+i)=(double **)malloc(matrix_rows_size*sizeof(double *));
        for(int j=0;j<matrix_rows_size;j++)
        {
            *(*(Weight_input+i)+j)=(double *)malloc(matrix_cols_size*sizeof(double));        
        } 
    }

    double ***padding_input;

    padding_input=(double ***)malloc(image_channel*sizeof(double **));
    
    for(int i=0;i<image_channel;i++)
    {
        *(padding_input+i)=(double **)malloc(matrix_rows_size*sizeof(double *));
        for(int j=0;j<matrix_rows_size;j++)
        {
            *(*(padding_input+i)+j)=(double *)malloc(matrix_cols_size*sizeof(double));        
        } 
    }

    double ***Filter_input;

    Filter_input=(double ***)malloc(filter*sizeof(double **));
    for(int i=0;i<image_channel;i++)
    {
        *(Filter_input+i)=(double **)malloc(filter*sizeof(double *));
        for(int j=0;j<filter;j++)
        {
            *(*(Filter_input+i)+j)=(double *)malloc(filter*sizeof(double));
        }
    }    

    double ***convolution_output;

    convolution_output=(double ***)malloc(image_channel*sizeof(double **));
    for(int i=0;i<image_channel;i++)
    {
        *(convolution_output+i)=(double **)malloc(convolution_rows_size*sizeof(double *));
        for(int j=0;j<convolution_rows_size;j++) 
        {
            *(*(convolution_output+i)+j)=(double *)malloc(convolution_cols_size*sizeof(double));
        }
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int ch=0;ch<image_channel;ch++)
    {
        for(int row=0;row<matrix_rows_size;row++)
        {
            for(int col=0;col<matrix_cols_size;col++)
            {
                Weight_input[ch][row][col] = 0;
            }
        }
    }

    for(int ch=0;ch<image_channel;ch++)
    {
        for(int row=0;row<image_rows;row++)
        {
            for(int col=0;col<image_cols;col++)
            {
                Weight_input[ch][row+padding][col+padding] = image[ch][row][col];
            }
        }
    }

    printf("Filter Input\n");
    for(int ch=0;ch<image_channel;ch++)
    {
        for(int filter_row=0;filter_row<filter;filter_row++)
        {
            for(int filter_col=0;filter_col<filter;filter_col++)
            {
                scanf("%lf",&Filter_input[ch][filter_row][filter_col]);
            }
        }
    }

    for(int ch=0;ch<image_channel;ch++)
    {
        for(int i=0;i<convolution_rows_size;i++)
        {
            for(int j=0;j<convolution_cols_size;j++)
            {
                convolution_output[ch][i][j] = 0;
            }
        }
    }

    for(int ch=0;ch<image_channel;ch++)
    {
        for(int i=0;i<convolution_rows_size;i++)
        {
            for(int j=0;j<convolution_cols_size;j++)
            {
                for(int k=0;k<filter;k++)
                {
                    for(int l=0;l<filter;l++)
                    {
                        convolution_output[ch][i][j] += Weight_input[ch][k+(i*stride)][l+(j*stride)] * Filter_input[ch][k][l];
                    }
                }

                if(convolution_output[ch][i][j] > 255) convolution_output[ch][i][j] = 255;
                else if(convolution_output[ch][i][j] < 0) convolution_output[ch][i][j] = 0;

            }
        }
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return convolution_output;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(int i=0;i<image_channel;i++) 
    {
        for(int j=0;j<image_rows;j++)
        {
            free(*(*(image+i)+j));
        }
        free(*(image+i));
    }

    for(int i=0;i<image_channel;i++) 
    {
        for(int j=0;j<filter;j++)
        {
            free(*(*(Filter_input+i)+j));
        }
        free(*(Filter_input+i));
    }

    for(int i=0;i<image_channel;i++) 
    {
        for(int j=0;j<convolution_rows_size;j++)
        {
            free(*(*(convolution_output+i)+j));
        }
        free(*(convolution_output+i));
    }
  
    free(image);
    free(Filter_input);
    free(convolution_output);
}


Mat conv(Mat input_image, int kernel_size)
{
    Mat output_image; 
    Mat kernel = (Mat_<float>(3,3) << 1,2,1, 2,4,2, 1,2,1);               // Edge Detect Filter
    filter2D(input_image, output_image, -1, kernel, Point(-1,-1), 0, 0);          //filter2D(input image, output image, image depth(CV_), Kernel, anchor, delta, BorderType(Padding))

    return output_image;
}


int main(void)
{
    Mat image;

    image = imread("index.jpg", IMREAD_COLOR);
    if(image.empty())
    {
        cout<<"Could not open or find the image"<<endl;
        return -1;
    }

    int filter = 0;
    printf("Filter Size: ");
    scanf("%d",&filter);

    int padding = 0;
    printf("PAD: ");
    scanf("%d",&padding);

    int stride=0;
    printf("Stride: ");
    scanf("%d",&stride);

    int matrix_rows_size = image.rows+(2*padding);
    int matrix_cols_size = image.cols+(2*padding);

    int convolution_rows_size = ((matrix_rows_size-filter)/stride) + 1;
    int convolution_cols_size = ((matrix_cols_size-filter)/stride) + 1;

    // Create Mat Class 
    Mat convolution_result(convolution_rows_size,convolution_cols_size,image.type());
    Mat convolution_2D_result(convolution_rows_size,convolution_cols_size,image.type());
    Mat convolution_compare(convolution_rows_size,convolution_cols_size,image.type());

    double ***jpg;

    jpg=(double ***)malloc(image.channels()*sizeof(double **));
    for(int i=0;i<image.channels();i++)
    {
        *(jpg+i)=(double **)malloc(image.rows*sizeof(double *));
        for(int j=0;j<image.rows;j++)
        {
            *(*(jpg+i)+j)=(double *)malloc(image.cols*sizeof(double));        
        } 
    }

    for(int i=0;i<image.channels();i++)
    {
        for(int j=0;j<image.rows;j++)
        {
            for(int k=0;k<image.cols;k++)
            {
                jpg[i][j][k]=image.at<cv::Vec3b>(j,k)[i];                                   // image -> 3 Dimension Pointer
            }
        }
    }

    double ***conv_jpg_out;

    conv_jpg_out=(double ***)malloc(convolution_result.channels()*sizeof(double **));
    for(int i=0;i<convolution_result.channels();i++)
    {
        *(conv_jpg_out+i)=(double **)malloc(convolution_result.rows*sizeof(double *));
        for(int j=0;j<convolution_result.rows;j++)
        {
            *(*(conv_jpg_out+i)+j)=(double *)malloc(convolution_result.cols*sizeof(double));        
        } 
    }

    printf("Image.Row: %d, Image.Col: %d\n",image.rows,image.cols);

    namedWindow("Original",WINDOW_AUTOSIZE);
    imshow("Original",image);

    conv_jpg_out = convolution(jpg, image.channels(), image.rows, image.cols, filter, padding, stride, matrix_rows_size, matrix_cols_size, convolution_rows_size, convolution_cols_size);
    for(int ch=0;ch<image.channels();ch++)
    {
        for(int i=0;i<convolution_result.rows;i++)
        {
            for(int j=0;j<convolution_result.cols;j++)
            {   
                convolution_result.at<Vec3b>(i,j)[ch]=conv_jpg_out[ch][i][j];                      // 3 Dimension Pointer Convolution Result -> Mat Convolution Result
            }
        }
    }

    convolution_2D_result = conv(image, filter);                                                   // Filter2D Function Result

    cv::subtract(convolution_2D_result,convolution_result,convolution_compare);                    // subtract(input image1, input image2, image1-image2)

    //printf("conv: %d filter2D: %d result: %d\n",convolution_result,convolution_2D_result,convolution_compare);
    cout << convolution_compare;
    namedWindow("Convolution",WINDOW_AUTOSIZE);
    imwrite("Convolution.jpg",convolution_compare);
    imshow("Convolution",convolution_compare);
    
    waitKey(0);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for(int i=0;i<image.channels();i++) 
    {
        for(int j=0;j<image.rows;j++)
        {
            free(*(*(jpg+i)+j));
        }
        free(*(jpg+i));
    }

    for(int i=0;i<convolution_result.channels();i++) 
    {
        for(int j=0;j<convolution_result.rows;j++)
        {
            free(*(*(conv_jpg_out+i)+j));
        }
        free(*(conv_jpg_out+i));
    }


    free(jpg);
    free(conv_jpg_out);

    return 0;
}