#include "imgui_fpsGraph.h"
#include "imgui.h"

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "imgui_internal.h"

#include <memory>
#include <algorithm>

namespace ImGui {
	
	void FPSGraph::setNumSamples(int n) {
		fps_num_samples = 20;
		times.resize(fps_num_samples);
		sortedTimes.resize(fps_num_samples);
	}

	FPSGraph::FPSGraph()
	{
		fps_num_samples = 20;
		setNumSamples(20);
		fps_min_time = 0.6f;
		fps_vertical_bars = 5;
		
		for (currentSample = 0; currentSample < fps_num_samples; ++currentSample)
		{
			times[currentSample] = 0.0f;
		}
		time = 0.0f;
		frames = 0;
		frame_size = ImVec2(100, 100);
		padding = 0;
		border = 8;
		margin = 0;

		//fill = QG::NONE;

		heightScale = 0.1f;//100ms

		tpfLast = 0.0f;
		fpsLast = 0.0f;
		tpfAverage = 0.0f;
		tpfMedian = 0.0f;
		tpfMaximum = 0.0f;
		tpfMinimum = 0.0f;

		print = false;
		gotNewSample = false;
	}

	void FPSGraph::updateStats()
	{
		memcpy(sortedTimes.data(), times.data(), fps_num_samples * sizeof(float));
		std::sort(sortedTimes.data(), sortedTimes.data() + fps_num_samples);

		tpfAverage = 0.0f;
		tpfMaximum = sortedTimes[0];
		tpfMinimum = sortedTimes[0];
		for (int i = 0; i < fps_num_samples; ++i)
		{
			tpfAverage += sortedTimes[i];
			if (tpfMaximum < sortedTimes[i]) tpfMaximum = sortedTimes[i];
			if (tpfMinimum > sortedTimes[i]) tpfMinimum = sortedTimes[i];
		}
		tpfAverage /= fps_num_samples;
		tpfMedian = sortedTimes[fps_num_samples / 2];
	}


	void FPSGraph::update(float dt)
	{
		time += dt;
		++frames;
		if (time > fps_min_time)
		{
			tpfLast = time / frames;
			fpsLast = frames / time;
			time = 0.0f;
			frames = 0;

			currentSample = (currentSample + 1) % fps_num_samples;
			times[currentSample] = tpfLast;

			updateStats();

			if (print)
				printf("%.2ffps %.2fms\n", 1.0 / tpfLast, 1000.0*tpfLast);

			gotNewSample = true;
		}
		else
			gotNewSample = false;

		if (tpfMaximum > 0)
		{
			float step = dt * 10000.0;
			float targetHeight = 0.6666 / tpfMaximum;
			float to = targetHeight - heightScale;
			if (fabs(to) > step)
				heightScale += step * to / fabs(to);
			else
				heightScale = targetHeight;
		}
	}

	bool FPSGraph::Plot(const char * label) {
		ImGuiWindow* window = GetCurrentWindow();
		if (window->SkipItems)
			return false;

		ImGuiContext & g = *GImGui;
		const ImGuiStyle& style = g.Style;
		const ImGuiID id = window->GetID(label);

		// draw background
		//const ImRect frame_bb(
		//	window->DC.CursorPos,
		//	window->DC.CursorPos + frame_size);
		const float w = CalcItemWidth();
		const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(w, frame_size.y + style.FramePadding.y * 2.0f));
		const ImRect inner_bb(
			frame_bb.Min + style.FramePadding,
			frame_bb.Max - style.FramePadding);
		const ImRect total_bb = frame_bb;
		ItemSize(total_bb, style.FramePadding.y);
		if (!ItemAdd(total_bb, 0, &frame_bb))
			return false;
		const bool hovered = ItemHoverable(frame_bb, id);

		RenderFrame(
			frame_bb.Min,
			frame_bb.Max,
			GetColorU32(ImGuiCol_FrameBg),
			true,
			style.FrameRounding);

		const float fps30 = 1.0 / 30.0;
		const float fps60 = 1.0 / 60.0;
		//float scalew = frame_size.x /*width*/ / (float)(fps_num_samples - 1);
		float ratew = 1.f / (float)(fps_num_samples - 1);

		float offsetx = -time / fps_min_time;
		ImU32 color;

		// graph grid
		const int vbarFreq = fps_num_samples / fps_vertical_bars;
		color = GetColorU32(ImVec4(0,1,0,0.5));
		float y0 = inner_bb.Min.y;
		float y1 = inner_bb.Max.y;
		for (int i = vbarFreq - currentSample % vbarFreq; i < fps_num_samples; i += vbarFreq)
		{
			float x = i + offsetx;
			if (x < 0.0f) x += 1.0f;
			float x0 = ImLerp(inner_bb.Min.x, inner_bb.Max.x, x * ratew);
			ImVec2 pos0 = ImVec2(x0, y0);
			ImVec2 pos1 = ImVec2(x0, y1);
			window->DrawList->AddLine(
				pos0,pos1,color,1.f
			);
		}
		// horizontal bars at 30 and 60 fps
		color = GetColorU32(ImVec4(0.8,0.2,0,0.5));
		if (fps30 * heightScale <= 1.0) {
			float t = 1.f - fps30 * heightScale;
			//float t = fps30 * heightScale;
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, t);
			ImVec2 pos0 = ImVec2(inner_bb.Min.x, y);
			ImVec2 pos1 = ImVec2(inner_bb.Max.x, y);
			window->DrawList->AddLine(
				pos0,pos1,color,1.f
			);
		}
		color = GetColorU32(ImVec4(0.2, 0.8, 0, 0.5));
		if (fps60 * heightScale <= 1.0) {
			float t = 1.f - fps60 * heightScale;
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, t);
			ImVec2 pos0 = ImVec2(inner_bb.Min.x, y);
			ImVec2 pos1 = ImVec2(inner_bb.Max.x, y);
			window->DrawList->AddLine(
				pos0, pos1, color, 1.f
			);
		}
		// graph line
		color = GetColorU32(ImVec4(0, 1, 0, 0.5));
		ImVec2 last,current;
		for (int i = 0; i < fps_num_samples; i++) {
			float t = times[(currentSample + 1 + i) % fps_num_samples];
			float x = i + offsetx;
			if (x < 0.0f) x = 0.0f;
			x = ImLerp(inner_bb.Min.x, inner_bb.Max.x, x * ratew);
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, 1 - t * heightScale);
			current = ImVec2(x, y);
			if (i > 0) {
				window->DrawList->AddLine(
					last, current, color, 1.f
				);
			}
			last = current;
		}

		// plot median and average
		if (tpfAverage * heightScale <= 1.0)
		{
			color = GetColorU32(ImVec4(0, 0.5, 1, 0.5));
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, 1.f - tpfAverage * heightScale);
			ImVec2 pos0 = ImVec2(inner_bb.Min.x, y);
			ImVec2 pos1 = ImVec2(inner_bb.Max.x, y);
			window->DrawList->AddLine(
				pos0, pos1, color, 1.f
			);
		}
		if (tpfMedian * heightScale <= 1.0)
		{
			color = GetColorU32(ImVec4(1, 0.5, 0.5, 0.5));
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, 1.f - tpfMedian * heightScale);
			ImVec2 pos0 = ImVec2(inner_bb.Min.x, y);
			ImVec2 pos1 = ImVec2(inner_bb.Max.x, y);
			window->DrawList->AddLine(
				pos0, pos1, color, 1.f
			);
		}
		// plot points
		color = GetColorU32(ImVec4(0, 1, 0, 1));
		for (int i = 0; i < fps_num_samples; i++) {
			float t = times[(currentSample + 1 + i) % fps_num_samples];
			if (t > fps30)
				color = GetColorU32(ImVec4(1, 0, 0, 1));
			else if (t > fps60)
				color = GetColorU32(ImVec4(0.8, 0.8, 0, 1));
			else
				color = GetColorU32(ImVec4(0, 1, 0, 1));
			float x = i + offsetx;
			if (x < 0.0f) x = 0.0f;
			x = ImLerp(inner_bb.Min.x, inner_bb.Max.x, x * ratew);
			float y = ImLerp(inner_bb.Min.y, inner_bb.Max.y, 1 - t * heightScale);
			current = ImVec2(x, y);
			//FIXME, it seems there is no draw point in ImGui, it is ugly...
			window->DrawList->AddCircle(current, 1.f, color);
		}
		// Text overlay
		if (overlay_text)
			RenderTextClipped(ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y), frame_bb.Max, overlay_text, NULL, NULL, ImVec2(0.5f, 0.0f));

		return true;
	}
	/*
	void FPSGraph::drawContent(mat44 mat)
	{
		mygl.projection = mat;

		const float fps30 = 1.0 / 30.0;
		const float fps60 = 1.0 / 60.0;
		glEnable(GL_BLEND);

		float scalew = width / (float)(GUIFPS_NUM_SAMPLES - 1);
		float offsetx = -time / GUIFPS_MIN_TIME;

		//graph grid
		mygl.colour(0, 1, 0, 0.3);
		mygl.begin(Immediate::LINES);
		const int vbarFreq = GUIFPS_NUM_SAMPLES / GUIFPS_VERTICAL_BARS;
		for (int i = vbarFreq - currentSample % vbarFreq; i < GUIFPS_NUM_SAMPLES; i += vbarFreq)
		{
			float x = i + offsetx;
			if (x < 0.0f) x += 1.0f;
			mygl.vertex(x*scalew, 0);
			mygl.vertex(x*scalew, height);
		}
		mygl.end(false);

		//horizontal bars at 30 and 60 fps
		mygl.colour(0, 1, 0, 0.5);
		mygl.begin(Immediate::LINES);
		if (fps30 * heightScale <= 1.0)
		{
			mygl.vertex(0, height - fps30 * heightScale * height);
			mygl.vertex(width, height - fps30 * heightScale * height);
		}
		if (fps60 * heightScale <= 1.0)
		{
			mygl.vertex(0, height - fps60 * heightScale * height);
			mygl.vertex(width, height - fps60 * heightScale * height);
		}
		mygl.end(false);

		//graph line
		mygl.colour(0, 1, 0, 0.5);
		mygl.begin(Immediate::LINE_STRIP);
		for (int i = 0; i < GUIFPS_NUM_SAMPLES; ++i)
		{
			float t = times[(currentSample + 1 + i) % GUIFPS_NUM_SAMPLES];
			float x = i + offsetx;
			if (x < 0.0f) x = 0.0f;
			mygl.vertex(x*scalew, height - t * heightScale * height);
		}
		mygl.end(false);

		//plot median and average
		mygl.begin(Immediate::LINES);
		if (tpfAverage * heightScale <= 1.0)
		{
			mygl.colour(0, 0.5, 1, 0.5);
			mygl.vertex(0, height - tpfAverage * heightScale * height);
			mygl.vertex(width, height - tpfAverage * heightScale * height);
		}
		if (tpfMedian * heightScale <= 1.0)
		{
			mygl.colour(1, 0.5, 0.5, 0.5);
			mygl.vertex(0, height - tpfMedian * heightScale * height);
			mygl.vertex(width, height - tpfMedian * heightScale * height);
		}
		mygl.end(false);

		//plot points
		mygl.colour(0, 1, 0, 1);
		mygl.begin(Immediate::POINTS);
		for (int i = 0; i < GUIFPS_NUM_SAMPLES; ++i)
		{
			float t = times[(currentSample + 1 + i) % GUIFPS_NUM_SAMPLES];
			if (t > fps30)
				mygl.colour(1, 0, 0, 1);
			else
				mygl.colour(0, 1, 0, 1);
			float x = i + offsetx;
			if (x < 0.0f) x = 0.0f;
			mygl.vertex(x*scalew, height - t * heightScale * height);
		}
		mygl.end(false);

		mygl.flush();
	}
	*/

}