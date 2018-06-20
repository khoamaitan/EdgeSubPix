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
		};

		enum Threshold {
			NB_OF_PTS = 0,
			LENGTH,
			AREA,
			ORIENTATION,
			NONE,
			NB_OF_THRESHOLDS
		};

		void extractEdges(int threshold);
		void extractContours(int threshold);
		void selectContour(int threshold);
		void selectAreaContours(int threshold);
		void selectLengthContours(int threshold);
		void selectNbOfPtsContours(int threshold);
		void selectOrientedContoursParts(int threshold);
		void filterContours(int threshold, void* type);
	
		bool nbPtsFilter(const sp::EdgesSubPix::Contour& contour, double threshold);
		bool lengthFilter(const sp::EdgesSubPix::Contour& contour, double threshold);
		bool areaFilter(const sp::EdgesSubPix::Contour& contour, double threshold);
		bool orientationPtFilter(const cv::Point2f& pt, cv::Point2f lineDir, double orientationTolerance, double angle_ref);
		cv::Vec4f contourOrientationLine(const std::vector< cv::Point2f >& pts);
		void filterContours(std::vector<sp::EdgesSubPix::Contour>& contours);

		void updateEdgesListFromROIs();
		void updateContoursListFromROIs();

		void resetExtraction();
		void resetAmbiguityImages();
		void setGrayImage(const cv::Mat& grayImage);
		void setEdgesImage(const cv::Mat& edges);
		void setEdgesMask(const cv::Mat& mask);
		void contourVec2VecForDisplay(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, std::vector < std::vector< cv::Point > >& contours, std::vector < cv::Vec4i >& hierarchy);
		void show(cv::Mat& image, const std::string& windowName);
		void setEdgesFromContours(const std::vector< sp::EdgesSubPix::Contour> & contoursPts, const cv::Mat& contours, cv::Mat& edges);
		void showEdgesFromContours(const std::vector< sp::EdgesSubPix::Contour> & contoursPts, const cv::Mat& contours, cv::Mat& edges, const std::string& windowName = "Edges from contours");
		void drawContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const cv::Scalar& color, bool markers = false, int markerType = cv::MarkerTypes::MARKER_CROSS, bool normals=false, int markersDisplayRatio=10, int normalsDisplayRatio=10, int contourThickness = 1);
		void drawContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage);
		void showContours(const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, bool markers = false, int markerType = cv::MarkerTypes::MARKER_CROSS, bool normals=false, int markersDisplayRatio=10, int normalsDisplayRatio=10, int contourThickness = 1);
        void showContour(const int& contourId, const std::vector<sp::EdgesSubPix::Contour>& contoursPts, cv::Mat& contoursImage, const std::string& windowName, const cv::Scalar& color, bool marker = false, int markerType = cv::MarkerTypes::MARKER_CROSS, bool normals=false, int markersDisplayRatio=10, int normalsDisplayRatio=10, bool selectContourStepByStep = false, int contourThickness = 1);
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
		std::string m_EDGES_FROM_CONTOURS = "Edges from contours";
        std::string m_CONTOURS_WINDOW_NAME = "Contours" ;
		std::string m_FILTERED_CONTOURS_WINDOW_NAME = "Filtered contours";
		std::string m_PIXEL_STATE_AFTER_EDGES_DETECTION_WINDOW_NAME = "Pixels state after edges detection";
		std::string m_PIXEL_STATE_AFTER_CONTOURS_DETECTION_WINDOW_NAME = "Pixels state after contours detection";
		std::string m_MOVING_EDGES_WINDOW_NAME = "Moving edges";
		std::string m_AMBIGUITY_ON_EDGES_IMAGE_LIST_WINDOW_NAME = "Ambiguity on edges in image list";
		std::string m_AMBIGUITY_ON_CONTOURS_IMAGE_LIST_WINDOW_NAME = "Ambiguity on contours in image list";

		cv::Mat m_image;
		cv::Mat m_edges;
		cv::Mat m_mask;
		cv::Mat m_contours;
		cv::Mat m_edgesFromContours;
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
		bool m_markers = false;
		int m_markerType = cv::MarkerTypes::MARKER_TILTED_CROSS;
		bool m_normals = false;
		bool m_selectContourStepByStep = false;
		int m_contourStep = 0;
		int m_selectedContourId = 0;
		int m_contourStepId = 0;

		bool m_filterNbOfPoints = true;
		bool m_filterLength = true;
		bool m_filterArea = true;
		bool m_filterOrientation = true;
		bool m_otherFilter = false;

		double m_length_threshold = 10;
		double m_nbOfPts_threshold = 10;
		double m_area_threshold = 10;
		double m_orientationTolerance = 30;
		double m_angle_ref = 0;


		const std::string INPUT_GRAYIMAGE = "Input gray image";
		const std::string EDGES_IMAGE = "Edges image";
		std::map< std::string, cv::Rect > m_rois;
		std::map< std::string, std::map< std::string, cv::Mat> > m_roiImages;

		std::vector<sp::EdgesSubPix::Edge> m_edgesInPixel;
		std::vector<sp::EdgesSubPix::Edge> m_edgesInSubPixel;
		std::vector<sp::EdgesSubPix::Contour> m_contoursPtsInSubPixel;
		std::vector<sp::EdgesSubPix::Contour> m_contoursPtsInPixel;

		std::map < std::string, std::vector < EdgesSubPix::Edge > > m_roisEdgesPts;
		std::map < std::string, std::vector < sp::EdgesSubPix::Contour> > m_roisContoursPts;
		std::map < std::string, std::vector < sp::EdgesSubPix::Contour> > m_roisContoursPtsFiltered;
		sp::EdgesSubPix::Contour m_selectedContour;
		sp::EdgesSubPix::Contour m_selectedContourInPixel;

		std::map < cv::String, std::map< int, std::vector<sp::EdgesSubPix::Edge> > > m_imageListEdgesAmbiguities;
		std::map < cv::String, std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > > m_imageListContoursAmbiguities;

		sp::EdgesSubPix m_edgesSubPix;
	};

}

#endif // __SUBPIX_H__
