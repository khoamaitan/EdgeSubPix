#ifndef __SUBPIX_H__
#define __SUBPIX_H__

// STL
#include <opencv2/opencv.hpp>

#include "EdgesSubPix.h"

namespace sp
{
    class SubPix
	{
	public:
		SubPix();
		~SubPix();

		enum PixelState {
			MULTICONTOUR=0,
			MULTISUBEDGES,
			TOTAL
		} ;

		void extractEdges(int threshold);
		void extractContours(int threshold);
		void selectContour(int threshold);
		void selectAreaContours(int threshold);

		void setGrayImage(const cv::Mat& grayImage);
		void setEdgesImage(const cv::Mat& edges);
		void setEdgesMask(const cv::Mat& mask);
		void contourVec2PtsVecVecForDisplay(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, std::vector < std::vector< cv::Point > >& contours);
		void showEdges(cv::Mat& edges, const std::string& windowName);
		void drawContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const cv::Scalar& color, bool markers = false);
		void drawContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::vector<cv::Vec4i>& hierarchy);
        void showContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, const std::vector<cv::Vec4i>& hierarchy, bool markers = false);
        void showContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, const cv::Scalar& color, bool marker = false);
		void destroyWindows();
		void parseInputPoints2YmlFile(const std::string& filename, std::vector<cv::Point>& inputPoints);
		void saveEdges2YmlFile(const std::string& filename, cv::Mat& edges);
		void saveImage(const std::string& filename, cv::Mat& image);
		void saveContours2YmlFile(const std::string& filename, const std::vector<sp::EdgesSubPix::Contour>& contours);
		void setROI(const std::string& roiName, const cv::Rect& roi);
		const cv::Rect getROI(const std::string& roiName);
		void setROIs(const std::vector<cv::Point>& inputPoints, const int& defaultSize = 20);
		void setROIs(const std::map< std::string, cv::Rect>& rois);
		void setImageROI(const std::string& roiName, const cv::Rect& roi);
		void getImageROI(const std::string& roiName, cv::Mat& imageROI, cv::Mat& edgesROI);
		void setImageROIs(const std::map< std::string, cv::Rect >& rois = std::map< std::string, cv::Rect >(), const cv::Mat& grayImage = cv::Mat());
		std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > pixelContoursMap(const std::vector<sp::EdgesSubPix::Contour>& contours, const std::vector<sp::EdgesSubPix::Contour>& subContours, int imageHeight);
		void computePixelState(std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >& pixelContourMap, std::map<int, std::map<int, bool> >& pixelState);
		cv::Mat displayPixelState(const std::map<int, std::map<int, bool> >& pixelState, int imageWidth, int imageHeight, const std::string& windowName = "Pixels state");
		std::map< int, std::vector<sp::EdgesSubPix::Edge> > pixelEdgesMap(const std::vector<sp::EdgesSubPix::Edge>& edges, const std::vector<sp::EdgesSubPix::Edge>& subEdges, int imageHeight);
		void computePixelState(std::map< int, std::vector<sp::EdgesSubPix::Edge> >& pixelEdgesMap, std::map<int, std::map<int, bool> >& pixelState);
		void saveEdges2YmlFile(const std::string& filename, const std::vector<sp::EdgesSubPix::Edge>& edges);
		void saveEdgesPixelMap(const std::string& filename, int imageHeight, std::map< int, std::vector<sp::EdgesSubPix::Edge > >& pixelEdgesMap);
		void saveContoursPixelMap(const std::string& filename, int imageHeight, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > >& pixelContoursMap);
		cv::Mat displayMovingEdges(const std::vector<sp::EdgesSubPix::Edge >& edgesInPixel, const std::vector<sp::EdgesSubPix::Edge >& edgesInSubPixel, int imageWidth, int imageHeight, const std::string& windowName = "Moving edges");
		cv::Mat displayMovingContourEdges(const std::vector<sp::EdgesSubPix::Contour>& contoursInPixel, const std::vector<sp::EdgesSubPix::Contour>& contoursInSubPixel, int imageWidth, int imageHeight, const std::string & windowName = "Moving edges");	
		cv::Mat displayImageSequenceEdgesAmbiguities(int imageWidth, int imageHeight, const std::string & windowName = "Ambiguity on edges in image list");
		cv::Mat displayImageSequenceContoursAmbiguities(int imageWidth, int imageHeight, const std::string & windowName = "Ambiguity on contours in image list");

        std::string m_IMAGE_WINDOW_NAME = "Image";
        std::string m_EDGES_WINDOW_NAME = "Edges";
        std::string m_CONTOURS_WINDOW_NAME = "Contours" ;
		std::string m_PIXEL_STATE_AFTER_EDGES_DETECTION_WINDOW_NAME = "Pixels state after edges detection";
		std::string m_PIXEL_STATE_AFTER_CONTOURS_DETECTION_WINDOW_NAME = "Pixels state after contours detection";
		std::string m_MOVING_EDGES_WINDOW_NAME = "Moving edges";
		std::string m_AMBIGUITY_ON_EDGES_IMAGE_LIST_WINDOW_NAME = "Ambiguity on edges in image list";
		std::string m_AMBIGUITY_ON_CONTOURS_IMAGE_LIST_WINDOW_NAME = "Ambiguity on contours in image list";

		cv::Mat m_image;
		cv::Mat m_edges;
		cv::Mat m_mask;
		cv::Mat m_contours;
		cv::String m_imageFile;
		cv::String m_inputPointsYmlFile;
		cv::String m_edgesFile;
		cv::String m_edgesImage;
		cv::String m_contoursImage;
        int m_low = 3;
		int m_high = 100;
		double m_alpha = 1.0;
		int m_contourMode = cv::RetrievalModes::RETR_EXTERNAL;
		std::vector<cv::Point> m_inputPoints;

		int m_nbOfEdges = 0;
		int m_nbOfContours = 0;
		int m_maxContours = 2000;
        bool m_display = false;

		const std::string INPUT_GRAYIMAGE = "Input gray image";
		const std::string EDGES_IMAGE = "Edges image";
		std::map< std::string, cv::Rect > m_rois;
		std::map< std::string, std::map< std::string, cv::Mat> > m_roiImages;

		std::vector<sp::EdgesSubPix::Edge> m_edgesInPixel;
		std::vector<sp::EdgesSubPix::Edge> m_edgesInSubPixel;
		std::vector<sp::EdgesSubPix::Contour> m_contoursPtsInSubPixel;
		std::vector<sp::EdgesSubPix::Contour> m_contoursPtsInPixel;
		std::vector<cv::Vec4i> m_hierarchy;

		std::map < std::string, std::vector < EdgesSubPix::Edge > > m_roisEdgesPts;
		std::map < std::string, std::vector < sp::EdgesSubPix::Contour> > m_roisContoursPts;
		std::map < std::string, std::vector<cv::Vec4i> > m_roisHierarchy;
		sp::EdgesSubPix::Contour m_selectedContour;
		sp::EdgesSubPix::Contour m_selectedContourInPixel;

		std::map < cv::String, std::map< int, std::vector<sp::EdgesSubPix::Edge> > > m_imageListEdgesAmbiguities;
		std::map < cv::String, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > > m_imageListContoursAmbiguities;

		sp::EdgesSubPix m_edgesSubPix;
	};

}

#endif // __SUBPIX_H__
