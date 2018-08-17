Sub-Pixel Edge Detection using OpenCV 
===

# Description

- Edge detection operator return subpixel level edge position.
- Using Canny filter as differential operator.
- Implementing according to [Carsten Steger's method][1].
- Interpolation of second partial derivatives with [Facet Model method][2].

# API

```cpp
namespace sp
{
	class EdgesSubPix
	{
		struct Contour
		{
			std::vector<cv::Point2f> points;  // edges locations
			std::vector<cv::Point> pointsInPix;  // edges locations in pixels
			std::vector<float> direction;     // direction of the gradients in contour edges points, 
											  // starting from y axis, counter-clockwise,
			std::vector<float> response;      // amplitude of the gradient in edges point,
			std::vector<float> nx;			  // amplitude of the gradient in x direction,
			std::vector<float> ny;			  // amplitude of the gradient in y direction
			double length;					  // length of the contour
			double area;					  // area of the contour
		};
		struct Edge
		{
			cv::Point2f point;				  // edge location
			cv::Point pointInPix;			  // edge location in pixel
			float direction;				  // direction of the gradients in edge point, 
											  // starting from y axis, counter-clockwise,
			float response;					  // amplitude of the gradient in edge point,
			float nx;						  // amplitude of the gradient in x direction,
			float ny;						  // amplitude of the gradient in y direction
		};
		// gray             - only support 8-bit grayscale
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixels, std::vector<Edge>& edgesInSubPixel, cv::Mat& edges);
		void contoursSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<Edge>& edgesInPixel, std::vector<Contour> &contoursInPixel,	std::vector<Contour> &contoursInSubPixel, int mode, cv::Mat& egdes);
	};
}
```

# License

3-clause BSD License

[1]:http://iuks.informatik.tu-muenchen.de/_media/members/steger/publications/1996/fgbv-96-03-steger.pdf
[2]:http://haralick.org/journals/topographic_primal_sketch.pdf
[3]:http://www2.thu.edu.tw/~emtools/OF/OF%20paper/Fast%20optical%20flow%20using%203D%20shortest%20path%20techniques.pdf

# Sources
https://github.com/songyuncen/EdgesSubPix