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
			std::vector<float> direction;     // direction of the gradients in contour edges points, 
											  // starting from y axis, counter-clockwise,
			std::vector<float> response;      // amplitude of the gradient in edges point,
			std::vector<float> nx;			  // amplitude of the gradient in x direction,
			std::vector<float> ny;			  // amplitude of the gradient in y direction
		};
		struct Edge
		{
			cv::Point2f point;				  // edge location
			float direction;				  // direction of the gradients in edge point, 
											  // starting from y axis, counter-clockwise,
			float response;					  // amplitude of the gradient in edge point,
			float nx;						  // amplitude of the gradient in x direction,
			float ny;						  // amplitude of the gradient in y direction
		};
		// gray             - only support 8-bit grayscale
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, std::vector<Edge>& edgesInPixels, std::vector<Edge>& edgesInSubPixel, cv::Mat& edges);

		// gray             - only support 8-bit grayscale
		// hierarchy, mode  - have the same meanings as in cv::findContours
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, std::vector<Edge>& edgesInPixel, std::vector<Contour> &contoursInPixel, std::vector<Contour> &contoursInSubPixel, cv::OutputArray hierarchy, int mode, cv::Mat& egdes);

		// mode = RETR_LIST	
		void edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, std::vector<Edge>& edgesInPixel, std::vector<Contour> &contoursInPixel, std::vector<Contour> &contoursInSubPixel);
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