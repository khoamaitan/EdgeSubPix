//Copyright(c) 2015, songyuncen / 2018, Fabien Contival
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met :
//
//*Redistributions of source code must retain the above copyright notice, this
//list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//this list of conditions and the following disclaimer in the documentation
//and / or other materials provided with the distribution.
//
//* Neither the name of EdgesSubPix nor the names of its
//contributors may be used to endorse or promote products derived from
//this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//	OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
		"{@movingEdgesImage |movingEdges.pgm | image for pixel moving edges after subpixel edges detection }"
		"{@pixelEdgesMapImage |pixelEdgesMap.pgm | image for pixel state after subpixel edges detection }"
		"{@contoursImage |contours.pgm | image for contours detection result }"
		"{@pixelContoursMapImage |pixelContoursMap.pgm | image for pixel state after subpixel contours detection }"
		"{contoursInPixelYmlFile |contoursInPixel.yml| file for contours in pixel          }"
		"{contoursInSubPixelYmlFile |contoursInSubPixel.yml| file for contours in subpixel         }"
		"{pixelContoursYmlFile |pixelEdgesMap.yml| file for pixel contours map         }"
		"{pixelContoursMapYmlFile |pixelContoursMap.yml| file for pixel contours map         }"
		"{low            |13            | low threshold                 }"
		"{high           |100           | high threshold                }"
		"{mode    |1     | edges : 0, contours : 1      }"
		"{contourMode    |1             | same as cv::findContours      }"
		"{alpha          |1.0           | gaussian alpha                }"
		"{@outputFolderPath   |results        | folder path for results                }"
		"{computeImageAmbiguity          |0           | compute image ambiguities (all image must be of the same size   }"
		"{@edgesAmbiguityImage |edgesAmbiguityImage.pgm | image for edges ambiguities between images of a same sequence }"
		"{@contoursAmbiguityImage |contoursAmbiguityImage.pgm | image for contours ambiguities between images of a same sequence }";

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

	for (int i = 0; i < (int)imageFiles.size(); i++) {
		
		std::stringstream ss;
		ss << i;
		cv::String imageFile = imageFiles[i];

		// set detector members
		detector.m_imageFile = imageFile;
		detector.m_inputPointsYmlFile = inputPointsYmlFile;

		////////////////////////////////////////////////////// Process /////////////////////////////////////////////////////

		// read input image
		detector.m_image = imread(imageFile, cv::IMREAD_GRAYSCALE);
		if (detector.m_image.empty()) {
			std::cout << "Cannot read input image..." << std::endl;
			return 0;
		}

		// create windows image
		cv::namedWindow(detector.m_IMAGE_WINDOW_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);

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
		//std::map< std::string, cv::Rect> rois;
		//rois["1"] = { cv::Rect(158, 40, 43, 40) };
		//rois["2"] = { cv::Rect(158, 120, 43, 40) };
		//detector.setROIs(rois);

		// create a Trackbar for user to enter threshold
		if (mode == 0) {

			// display trackbar for edges detection
			if (detector.m_display) {
				cv::namedWindow(detector.m_EDGES_WINDOW_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
				cv::createTrackbar("Min Threshold:", detector.m_EDGES_WINDOW_NAME, &low, high, extractEdges);
			}

			// launch edges detection
			extractEdges(low, 0);
		}
		else
		{
			// display trackbar for contours detection
			if (detector.m_display) {
				cv::namedWindow(detector.m_EDGES_WINDOW_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
				cv::namedWindow(detector.m_CONTOURS_WINDOW_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
				cv::createTrackbar("Min Threshold:", detector.m_EDGES_WINDOW_NAME, &low, high, extractContours);
			}

			// launch contours detection
			extractContours(low, 0);

			// display trackbar for contours selection
			if (detector.m_display) {
				int firstContour = 0;
				int lastContour = std::max(firstContour, (int)detector.m_nbOfContours - 1);
				cv::createTrackbar("Contour Selector:", detector.m_CONTOURS_WINDOW_NAME, &firstContour, lastContour, contourSelector);
			}
		}
		cv::waitKey(0);

		// destroy detector windows
		detector.destroyWindows();
		cv::destroyWindow(detector.m_IMAGE_WINDOW_NAME);

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
