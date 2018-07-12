#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>
#include "SubPix.h"



sp::SubPix detector;


void extractEdges(int threshold, void*)
{
	detector.extractEdges(threshold);
}

void extractContours(int threshold, void*)
{
	detector.extractContours(threshold);
}

void contourSelector(int threshold, void*)
{	
	detector.selectContour(threshold);
}

void contourAreaSelector(int threshold, void*)
{
	detector.selectAreaContours(threshold);
}

void contourLengthSelector(int threshold, void*)
{
	detector.selectLengthContours(threshold);
}

void contourNbOfPtsSelector(int threshold, void*)
{
	detector.selectNbOfPtsContours(threshold);
}

void contourOrientationPtsSelector(int threshold, void*)
{
	detector.selectOrientedContoursParts(threshold);
}

void filterContoursSelector(int threshold, void* type)
{
	detector.filterContours(threshold, type);
}

std::vector < std::string > split(cv::String imageList)
{
	std::string imageList_no_separator = std::string(imageList);
	std::replace(imageList_no_separator.begin(), imageList_no_separator.end(), '#', ' ');
	std::istringstream imageListSS(imageList_no_separator);
	
	std::vector < std::string > out;
	std::copy(std::istream_iterator<std::string>(imageListSS), std::istream_iterator<std::string>(), std::back_inserter< std::vector< std::string > >(out));

	return out;
}

int main(int argc, char *argv[])
{
	cv::ocl::setUseOpenCL(true);
	const cv::String keys =
		"{help h usage ? |              | print this message            }"
		"{@imageFilesList         |              | images for edge detection using # for separation     }"
		"{inputPointsYmlFile |			    | pt to find subpixel location   }"
		"{@edgesImage |edges.pgm | image for edge detection result }"
		"{edgesInPixelYmlFile |edgesInPixel.yml| file for edges in pixel         }"
		"{edgesInSubPixelYmlFile |edgesInSubPixel.yml| file for edges in subpixel         }"
		"{pixelEdgesMapYmlFile |pixelEdgesMap.yml| file for pixel edges map         }"
		"{@movingEdgesImage |movingEdges.ppm | image for pixel moving edges after subpixel edges detection }"
		"{@pixelEdgesMapImage |pixelEdgesMap.ppm | image for pixel state after subpixel edges detection }"
		"{@contoursImage |contours.ppm | image for contours detection result }"
		"{@pixelContoursMapImage |pixelContoursMap.ppm | image for pixel state after subpixel contours detection }"
		"{contoursInPixelYmlFile |contoursInPixel.yml| file for contours in pixel          }"
		"{contoursInSubPixelYmlFile |contoursInSubPixel.yml| file for contours in subpixel         }"
		"{pixelContoursYmlFile |pixelEdgesMap.yml| file for pixel contours map         }"
		"{pixelContoursMapYmlFile |pixelContoursMap.yml| file for pixel contours map         }"
		"{low            |13            | low threshold                 }"
		"{high           |100           | high threshold                }"
		"{mode    |1     | edges : 0, contours : 1      }"
		"{contourMode    |0             | same as cv::findContours      }"
		"{alpha          |1.0           | gaussian alpha                }"
		"{@outputFolderPath   |results        | folder path for results                }"
		"{computeImageAmbiguity          |true           | compute image ambiguities (all image must be of the same size   }"
		"{selectContourStepByStep          |false           | select the contour step by step mode  }"
		"{filterContours          |true           | activate contours filtering }"
		"{@edgesAmbiguityImage |edgesAmbiguityImage.ppm | image for edges ambiguities between images of a same sequence }"
		"{@contoursAmbiguityImage |contoursAmbiguityImage.ppm | image for contours ambiguities between images of a same sequence }";

	// parse inputs
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Subpixel edge detection");

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	if (!parser.has("@imageFilesList"))
	{
		parser.printMessage();
		return 0;
	}

	cv::String imageFilesList = parser.get<cv::String>("@imageFilesList");
	cv::String edgesImage = parser.get<cv::String>("@edgesImage");
	cv::String inputPointsYmlFile = parser.get<cv::String>("inputPointsYmlFile");
	cv::String edgesInPixelYmlFile = parser.get<cv::String>("edgesInPixelYmlFile");
	cv::String edgesInSubPixelYmlFile = parser.get<cv::String>("edgesInSubPixelYmlFile");
	cv::String pixelEdgesMapYmlFile = parser.get<cv::String>("pixelEdgesMapYmlFile");
	cv::String movingEdgesImage = parser.get<cv::String>("@movingEdgesImage");
	cv::String pixelEdgesMapImage = parser.get<cv::String>("@pixelEdgesMapImage");

	cv::String contoursImage = parser.get<cv::String>("@contoursImage");
	cv::String contoursInPixelYmlFile = parser.get<cv::String>("contoursInPixelYmlFile");
	cv::String contoursInSubPixelYmlFile = parser.get<cv::String>("contoursInSubPixelYmlFile");
	cv::String pixelContoursMapYmlFile = parser.get<cv::String>("pixelContoursMapYmlFile");
	cv::String pixelContoursMapImage = parser.get<cv::String>("@pixelContoursMapImage");
	bool computeImageAmbiguity = parser.get<bool>("computeImageAmbiguity");
	bool selectContourStepByStep = parser.get<bool>("selectContourStepByStep");
	bool filterContours = parser.get<bool>("filterContours");
	cv::String outputFolderPath = parser.get<cv::String>("@outputFolderPath");
	cv::String edgesAmbiguityImage = parser.get<cv::String>("@edgesAmbiguityImage");
	cv::String contoursAmbiguityImage = parser.get<cv::String>("@contoursAmbiguityImage");

	int low = parser.get<int>("low");
	int high = parser.get<int>("high");
	double alpha = parser.get<double>("alpha");
	int mode = parser.get<int>("mode");
	int contourMode = parser.get<int>("contourMode");
	detector.m_contoursImage = contoursImage;
	detector.m_low = low;
	detector.m_high = high;
	detector.m_alpha = alpha;
	detector.m_contourMode = contourMode;
	detector.m_display = true;
	
	// get images
	std::vector< std::string > imageFiles = split(imageFilesList);

	// create folder for results
	std::string removeFolderCommand = "rmdir /Q /S " + outputFolderPath;
	system(removeFolderCommand.c_str());
	std::string createFolderCommand = "mkdir " + outputFolderPath;
	system(createFolderCommand.c_str());

	for (int i = 0; i < (int)imageFiles.size(); i++) 
	{	
		std::stringstream ss;
		ss << i;
		cv::String imageFile = imageFiles[i];

		// reset extraction
		detector.resetExtraction();

		// set detector members
		detector.m_imageFile = imageFile;
		detector.m_inputPointsYmlFile = inputPointsYmlFile;
		int firstContour = 0;
		int lastContour = 0;
		int area = 0;
		int length = 0;
		int nbOfPoints = 0;
		int orient = 0;

		////////////////////////////////////////////////////// Process /////////////////////////////////////////////////////

		// read input image
		detector.m_image = imread(imageFile, cv::IMREAD_GRAYSCALE);

#if TEST_PRE_FILTER
		cv::Mat framePatternSmooth;
		cv::bilateralFilter(detector.m_image, framePatternSmooth, 5, 50, 50);
		detector.m_image = framePatternSmooth;
		detector.m_alpha = 0.0;
#endif

		if (detector.m_image.empty()) {
			std::cout << "Cannot read input image..." << std::endl;
			return 0;
		}

		// create windows image
		cv::namedWindow(detector.m_IMAGE_WINDOW_NAME, cv::WINDOW_GUI_EXPANDED);
		cv::resizeWindow(detector.m_IMAGE_WINDOW_NAME, detector.m_image.cols, detector.m_image.rows);

		// show input image
		cv::imshow(detector.m_IMAGE_WINDOW_NAME, detector.m_image);

		// save edges image
		detector.saveImage(outputFolderPath + "/" + "input.pgm", detector.m_image);

		// parse input points from yaml file
		std::vector<cv::Point> inputPoints;
		detector.parseInputPoints2YmlFile(inputPointsYmlFile, inputPoints);
		detector.m_inputPoints = inputPoints;

		// set input ROI from input points
		detector.setROIs(inputPoints);

		// test some regions mannually
#if ROI_TEST
		std::map< std::string, cv::Rect> rois;
		rois["1"] = { cv::Rect(158, 40, 43, 40) };
		rois["2"] = { cv::Rect(158, 120, 43, 40) };
		detector.setROIs(rois);
#endif

		// create a Trackbar for user to enter threshold
		if (mode == 0) {

			// display trackbar for edges detection
			if (detector.m_display) {
				cv::namedWindow(detector.m_EDGES_WINDOW_NAME, cv::WINDOW_GUI_EXPANDED);
				cv::resizeWindow(detector.m_EDGES_WINDOW_NAME, detector.m_image.cols, detector.m_image.rows);
				cv::createTrackbar("Min:", detector.m_EDGES_WINDOW_NAME, &low, high, extractEdges);
			}

			// launch edges detection
			extractEdges(low, 0);
		}
		else
		{
			// display trackbar for contours detection
			if (detector.m_display) {
				cv::namedWindow(detector.m_EDGES_WINDOW_NAME, cv::WINDOW_GUI_EXPANDED);
				cv::namedWindow(detector.m_CONTOURS_WINDOW_NAME, cv::WINDOW_GUI_EXPANDED);
				cv::resizeWindow(detector.m_EDGES_WINDOW_NAME, detector.m_image.cols, detector.m_image.rows);
				cv::resizeWindow(detector.m_CONTOURS_WINDOW_NAME, detector.m_image.cols, detector.m_image.rows);
				cv::createTrackbar("Min:", detector.m_EDGES_WINDOW_NAME, &low, high, extractContours);
			}

			// launch contours detection
			extractContours(low, 0);

			// display trackbar for contours selection
			if (detector.m_display) {
				lastContour = std::max(firstContour, (int)detector.m_nbOfContours - 1);
				cv::createTrackbar("Contour:", detector.m_CONTOURS_WINDOW_NAME, &firstContour, lastContour, contourSelector);
				cv::createTrackbar("Area:", detector.m_CONTOURS_WINDOW_NAME, &area, 1000, contourAreaSelector);
				cv::createTrackbar("Length:", detector.m_CONTOURS_WINDOW_NAME, &length, 1000, contourLengthSelector);
				cv::createTrackbar("Points:", detector.m_CONTOURS_WINDOW_NAME, &nbOfPoints, 1000, contourNbOfPtsSelector);
				cv::createTrackbar("Orient:", detector.m_CONTOURS_WINDOW_NAME, &orient, 1000, contourOrientationPtsSelector);

				if (filterContours) {
					cv::namedWindow(detector.m_FILTERED_CONTOURS_WINDOW_NAME, cv::WINDOW_GUI_EXPANDED);
					cv::resizeWindow(detector.m_FILTERED_CONTOURS_WINDOW_NAME, detector.m_image.cols, detector.m_image.rows);

					int thresholdType[sp::SubPix::Threshold::NB_OF_THRESHOLDS];
					thresholdType[sp::SubPix::Threshold::AREA] = sp::SubPix::Threshold::AREA;
					cv::createTrackbar("Area:", detector.m_FILTERED_CONTOURS_WINDOW_NAME, &area, 1000, filterContoursSelector, (void*)&thresholdType[sp::SubPix::Threshold::AREA]);
					thresholdType[sp::SubPix::Threshold::LENGTH] = sp::SubPix::Threshold::LENGTH;
					cv::createTrackbar("Length:", detector.m_FILTERED_CONTOURS_WINDOW_NAME, &length, 1000, filterContoursSelector, (void*)&thresholdType[sp::SubPix::Threshold::LENGTH]);
					thresholdType[sp::SubPix::Threshold::NB_OF_PTS] = sp::SubPix::Threshold::NB_OF_PTS;
					cv::createTrackbar("Points:", detector.m_FILTERED_CONTOURS_WINDOW_NAME, &nbOfPoints, 1000, filterContoursSelector, (void*)&thresholdType[sp::SubPix::Threshold::NB_OF_PTS]);
					thresholdType[sp::SubPix::Threshold::ORIENTATION] = sp::SubPix::Threshold::ORIENTATION;
					cv::createTrackbar("Orient:", detector.m_FILTERED_CONTOURS_WINDOW_NAME, &orient, 1000, filterContoursSelector, (void*)&thresholdType[sp::SubPix::Threshold::ORIENTATION]);
					thresholdType[sp::SubPix::Threshold::NONE] = sp::SubPix::Threshold::NONE;
					filterContoursSelector(0, (void*)&thresholdType[sp::SubPix::Threshold::NONE]);
				}
			}
		}
		
		cv::waitKey(0);

		// update pts from filter output
		if (filterContours) {
			detector.updateContoursListFromROIs();
		}
		detector.m_selectContourStepByStep = selectContourStepByStep;

		while (detector.m_selectContourStepByStep)
		{
			std::cout << firstContour << std::endl;
			contourSelector(firstContour, 0);
			cv::waitKey(0);
		}

		// destroy detector windows
		detector.destroyWindows();

		////////////////////////////////////////////////////// Results /////////////////////////////////////////////////////

		// compute pixel state
		std::map<int, std::map<int, bool> > pixelState;
		std::map< int, std::vector<sp::EdgesSubPix::Edge> > pixelEdgesMap;
		cv::Mat pixelStateImage;
		cv::Mat movingEdges;
		std::map< int, std::map<int, std::vector<sp::EdgesSubPix::Edge> > > pixelContourMap;

		if (mode == 0) {

			// compute pixel state after subpixel edges detection
			std::vector<sp::EdgesSubPix::Edge> edgesPts;
			edgesPts = detector.m_edgesInSubPixel;
			std::vector<sp::EdgesSubPix::Edge> edgesInPixelPts;
			edgesInPixelPts = detector.m_edgesInPixel;
			pixelEdgesMap = detector.pixelEdgesMap(edgesInPixelPts, edgesPts, detector.m_image.rows);
			detector.computePixelState(pixelEdgesMap, pixelState);
			pixelStateImage = detector.displayPixelState(pixelState, detector.m_image.cols, detector.m_image.rows, detector.m_PIXEL_STATE_AFTER_EDGES_DETECTION_WINDOW_NAME);
			movingEdges = detector.displayMovingEdges(edgesInPixelPts, edgesPts, detector.m_image.cols, detector.m_image.rows);
		}
		else
		{
			// compute pixel state after subpixel contours detection
			std::vector<sp::EdgesSubPix::Contour> contoursPts;
			std::vector<sp::EdgesSubPix::Contour> contoursInPixelPts;
			if (!detector.m_selectedContour.points.empty()) {
				contoursPts.push_back(detector.m_selectedContour);
				contoursInPixelPts.push_back(detector.m_selectedContourInPixel);
			}
			else {
				contoursPts = detector.m_contoursPtsInSubPixel;
				contoursInPixelPts = detector.m_contoursPtsInPixel;
			}
			pixelContourMap = detector.pixelContoursMap(contoursInPixelPts, contoursPts, detector.m_image.rows);
			detector.computePixelState(pixelContourMap, pixelState);
			pixelStateImage = detector.displayPixelState(pixelState, detector.m_image.cols, detector.m_image.rows, detector.m_PIXEL_STATE_AFTER_CONTOURS_DETECTION_WINDOW_NAME);
			movingEdges = detector.displayMovingContourEdges(contoursInPixelPts, contoursPts, detector.m_image.cols, detector.m_image.rows);
		}
		cv::waitKey(0);
		detector.destroyWindows();

		// save results
		std::cout << "Saving results" << std::endl;

		// save edges image
		cv::Mat edges = detector.m_edges;
		detector.saveImage(outputFolderPath + "/" + ss.str() + "_" + edgesImage, edges);

		// save edges points to yml file
		std::vector<sp::EdgesSubPix::Edge> edgesInPixelPts = detector.m_edgesInPixel;
		detector.saveEdges2YmlFile(outputFolderPath + "/" + ss.str() + "_" + edgesInPixelYmlFile, edgesInPixelPts);

		// save subpixel edges points to yml file
		std::vector<sp::EdgesSubPix::Edge> edgesPts = detector.m_edgesInSubPixel;
		detector.saveEdges2YmlFile(outputFolderPath + "/" + ss.str() + "_" + edgesInSubPixelYmlFile, edgesPts);

		// save moving edges image
		detector.saveImage(outputFolderPath + "/" + ss.str() + "_" + movingEdgesImage, movingEdges);

		// save pixel state image
		detector.saveImage(outputFolderPath + "/" + ss.str() + "_" + pixelEdgesMapImage, pixelStateImage);

		// save pixel edges map
		detector.saveEdgesPixelMap(outputFolderPath + "/" + ss.str() + "_" + pixelEdgesMapYmlFile, detector.m_image.rows, pixelEdgesMap);

		// save contours image
		cv::Mat contours = detector.m_contours;
		detector.saveImage(outputFolderPath + "/" + ss.str() + "_" + contoursImage, contours);

		// save pixel contours to yml file
		std::vector< sp::EdgesSubPix::Contour > contoursInPixelPts = detector.m_contoursPtsInPixel;
		detector.saveContours2YmlFile(outputFolderPath + "/" + ss.str() + "_" + contoursInPixelYmlFile, contoursInPixelPts);

		// save subpixel contours to yml file
		std::vector<sp::EdgesSubPix::Contour> contoursInSubpixelPts = detector.m_contoursPtsInSubPixel;
		detector.saveContours2YmlFile(outputFolderPath + "/" + ss.str() + "_" + contoursInSubPixelYmlFile, contoursInSubpixelPts);

		// save pixel state image
		detector.saveImage(outputFolderPath + "/" + pixelContoursMapImage, pixelStateImage);

		// save pixel contours map
		detector.saveContoursPixelMap(outputFolderPath + "/" + ss.str() + "_" + pixelContoursMapYmlFile, detector.m_image.rows, pixelContourMap);

		// store image pixel map
		if (mode == 0) {
			detector.m_imageListEdgesAmbiguities[imageFile] = pixelEdgesMap;
		}
		else
		{
			detector.m_imageListContoursAmbiguities[imageFile] = pixelContourMap;
		}
	}

	// compute ambiguity maps
	if (computeImageAmbiguity) {

		cv::Mat ambiguityImage;
		if (mode == 0) {
			ambiguityImage = detector.displayImageSequenceEdgesAmbiguities(detector.m_image.cols, detector.m_image.rows, "Ambiguity on edges in image list");
			detector.saveImage(outputFolderPath + "/" + edgesAmbiguityImage, ambiguityImage);
		}
		else
		{
			ambiguityImage = detector.displayImageSequenceContoursAmbiguities(detector.m_image.cols, detector.m_image.rows, "Ambiguity on contours in image list");
			detector.saveImage(outputFolderPath + "/" + contoursAmbiguityImage, ambiguityImage);
		}
		cv::waitKey(0);
	}

    return 0;
}
