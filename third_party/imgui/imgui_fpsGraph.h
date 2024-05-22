#pragma once
// msj:modified from guifps.h/guifps.cpp
// 2021/4/25

#include "imgui.h"
#include <vector>

namespace ImGui {

	struct FPSGraph {

		int fps_num_samples;
		float fps_min_time;
		int fps_vertical_bars;

		//int width, height;
		int padding;
		int border;
		int margin;
		ImVec2 frame_size;
		const char * overlay_text = nullptr;
		
		//
		bool gotNewSample;
		bool print;
		float time;
		float frames;
		float fpsLast;
		float tpfLast;
		float tpfAverage;
		float tpfMedian;
		float tpfMaximum;
		float tpfMinimum;
		float heightScale;
		std::vector<float> times;
		std::vector<float> sortedTimes;
		//float times[GUIFPS_NUM_SAMPLES]; //cyclical list
		int currentSample; 

		FPSGraph();
		void updateStats();
		void update(float dt);
		void setNumSamples(int n); // call me before the pipeline start !

		//virtual void update();
		bool Plot(const char * label);

	};

	

	

}