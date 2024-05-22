#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "utils/ShaderManager.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <stdexcept>

#include "argh.h"

#include "utils/rot.h"
#include "utils/DoubleClick.h"
#include "utils/Mesh.h"

#include "cuGaussianSplatting/cuGaussianSplatting.h"
#include "cuGaussianSplatting/rasterizer.h"

#include "gsutils/containers.h"
#include "gsutils/operators.h"
#include "gsutils/gs_face.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/ImGuiFileDialog.h"


#include <unordered_set>

#ifndef M_PI
#define M_PI 3.1415926535897932384626
#endif

#define ENABLE_VSYNC

#define ENABLE_MSAA
#define MSAA_SAMPLE_NUM 4

#define FPS_GRAPH

namespace ImGui {
	void ShowDemoWindow(bool* p_open);
}

namespace {

	std::string g_model_path;
	std::string g_load_iter = "40000";
	std::string g_dataset_path;

	int n_test = 350; // keep last ? frames for testing
	int winWidth = 1280, winHeight = 720;
	float g_fovy = 45.f; float ui_fovy = g_fovy;
	//const float zoom_init = -1.f;
	float zoom_init = -1.f;
	float zoom = zoom_init;
	RotTools rotter; glutDoubleClick gdc;
	glutTimeInterval gti_rotate;
	glutTimeInterval gti_video; int play_framerate = 25;
	//glutTimeInterval gti_update;

	gaussian_splatting::gl_buffer_object globj;
	
	int model_type = 0;
	gaussian_splatting::transform_v1 loc_transfer;
	float hyper_param_k = 0.01f;

	//// Gaussian Splatting
	
	cuGaussianSplatting::GPoints cuPoints;
	cuGaussianSplatting::Transforms cuTrans;
	size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
	void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
	std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;
	gaussian_splatting::gl_buffer_object* cuBuffer = nullptr;

	ShaderManager gb_shader;

	// CPU objects
	gaussian_splatting::gaussian_face gs_face;
	gaussian_splatting::gaussian_basic gs_mouth0;
	gaussian_splatting::gaussian_basic gs_mouth1;

	// GPU objects
	cuGaussianSplatting::cuGaussianFace cu_face;
	cuGaussianSplatting::cuGaussianBasic cu_mouth0;
	cuGaussianSplatting::cuGaussianBasic cu_mouth1;

	int g_face_offset; 
	int g_mouth0_offset;
	int g_mouth1_offset;
	
	bool enable_deform_rot; bool enable_deform_rot_sh;
	bool enable_trans_rot; bool enable_trans_rot_sh;
	bool enable_mouth_rot; bool enable_mouth_rot_sh;

	bool ui_deform_rot; bool ui_deform_rot_sh;
	bool ui_trans_rot; bool ui_trans_rot_sh;
	bool ui_mouth_rot; bool ui_mouth_rot_sh;


	//// FLAME
	//gaussian_splatting::TorchFrame* torch_frame = nullptr;
	std::vector<gaussian_splatting::TorchFrame> torch_frames; 

	gaussian_splatting::FaceParams faceParams;
	gaussian_splatting::FaceParams zeroFaceParams;

	cuGaussianSplatting::cuBuffer cuParams;

	SMesh g_face_mesh;

	gaussian_splatting::FLAME g_flame;
		
	bool ignoring = false;

	bool left_down = false, right_down = false;
	bool enable_rot = false;
	bool enable_play = false;
	bool enable_play_camera = true;
	bool enable_fix_camera = false;

	// vis camera
	gaussian_splatting::CameraVisualizer* camera_vis = nullptr;
	int sel_camera = 0;
	int ui_sel_camera = 0;

	//// gui
	float point_size = 4.f;
	float point_color_att = 0.6f;
	float gsp_scalingModifer = 1.f;

	float gsp_offsetModifier = 1.f; float ui_gsp_offsetModifier = 1.f;

	float max_show_anistropy = 50.f;
	float scale_show_depth = 8.f;

	//int render_mode = 0;
	int render_mode = 2;
	int render_part_code = 0;
	bool should_update_mesh = false; // update mesh flag
	bool should_update_gs = false; // update gaussian flag
	bool should_update_bkg = false;
	bool should_update_frame = false;
	float g_bkg_color[3] = { 0,0,0 };
	float ui_bkg_color[3] = { 0,0,0 };
	
	bool use_msaa;
	bool display_obj = true;
	bool display_axis = true;
	//bool display_cam = true;
	bool display_cam = false;

	bool show_face = true;
	bool is_face_smooth = true;
	bool show_edge = false;

	// record
	bool record_flag = false;
	int record_n = 0;
	double record_time1 = 0;
	glutTimeInterval gti_record;
	

	enum {
		PC_COLOR, PC_COLOR_QUAD
	};

	void error_callback(int error, const char* description)
	{
		fprintf(stderr, "Error: %s\n", description);
	}

	void loadFrameSeq();
	void loadInfoFromFrame(gaussian_splatting::TorchFrame* frame);

	void print_time(int time_in_ms, int n_frames) {

		double time = double(time_in_ms) / 1000.0; // ms -> s
		double fps = double(n_frames) / time;
		double frame_per_ms = double(time_in_ms) / double(n_frames);
		printf("elapsed time %f s, %f ms per frame, fps %f\n", time, frame_per_ms, fps);
	}

	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		else if (key == GLFW_KEY_Q && action == GLFW_PRESS) 
		{
			if (record_flag == false) {
				record_flag = true;
				record_n = 0;
				record_time1 = 0;
				printf("start record\n");
				gti_record.touch();
			}
			else {
				record_flag = false;
				int time_in_ms = gti_record.touch();
				print_time(time_in_ms, record_n);
				{
					double frame_per_ms = double(record_time1) * 1000.0 / double(record_n);
					printf("elapsed time for cpu compute mesh %f ms\n", frame_per_ms);
				}
				//print_time(time_in_ms, record_n2);
			}

		}
		else if (key == GLFW_KEY_R && action == GLFW_PRESS)
		{
			enable_rot = !enable_rot;
			gti_rotate.touch();
		}
		else if (key == GLFW_KEY_P && action == GLFW_PRESS)
		{
			enable_play = !enable_play;
			gti_video.touch();		
		}
		else if (key == GLFW_KEY_C && action == GLFW_PRESS)
		{
			display_axis = !display_axis;
		}
		else if (key == GLFW_KEY_O && action == GLFW_PRESS)
		{
			display_obj = !display_obj;
		}
		else if (key == GLFW_KEY_T && action == GLFW_PRESS) 
		{
			display_cam = !display_cam;
		}
		else if (key == GLFW_KEY_F && action == GLFW_PRESS) 
		{
			show_face = !show_face;		
		}
		else if (key == GLFW_KEY_S && action == GLFW_PRESS) 
		{
			is_face_smooth = !is_face_smooth;
		}
		else if (key == GLFW_KEY_E && action == GLFW_PRESS) 
		{
			show_edge = !show_edge;
		}

		else if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
			render_mode = 0;
		}
		else if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
			render_mode = 1;
		}
		else if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
			render_mode = 2;
		}
		else if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
			render_mode = 3;
		}
		else if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
			render_mode = 4;
		}
		else if (key == GLFW_KEY_6 && action == GLFW_PRESS) {
			render_mode = 5;
		}
		else if (camera_vis) {
			if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
				sel_camera = (sel_camera + 1) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
				sel_camera = (sel_camera + 10) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
				sel_camera = (sel_camera + camera_vis->size() - 1) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
				sel_camera = (sel_camera + camera_vis->size() - 10) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_N && action == GLFW_PRESS) {
				sel_camera = (sel_camera + camera_vis->size() + 100) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_M && action == GLFW_PRESS) {
				sel_camera = (sel_camera + camera_vis->size() - 100) % camera_vis->size();
				ui_sel_camera = sel_camera;
				should_update_frame = true;
			}
			else if (key == GLFW_KEY_I && action == GLFW_PRESS) {
				printf("sel_camera: %d, coord(%f,%f,%f)\n", sel_camera,
					camera_vis->v[sel_camera].c2w[9],
					camera_vis->v[sel_camera].c2w[10],
					camera_vis->v[sel_camera].c2w[11]
				);
			}
		}
	}


	void load_back_head_index(const char * filename) {

		std::vector<int>& back_head_vertex = loc_transfer.back_head_vertex;
		std::vector<int> back_head_face_index;
		std::unordered_set<int> unrepeat_vertex;
		gaussian_splatting::get_index(back_head_face_index, filename);
		for (int i = 0; i < back_head_face_index.size(); i++) {
			SMesh::obj_face& face = g_face_mesh.face_list[back_head_face_index[i]];
			unrepeat_vertex.insert(face.vertex_index[0]);
			unrepeat_vertex.insert(face.vertex_index[1]);
			unrepeat_vertex.insert(face.vertex_index[2]);
		}
		back_head_vertex.clear();
		for (auto it = unrepeat_vertex.begin(); it != unrepeat_vertex.end(); it++) {
			back_head_vertex.push_back(*it);
		}
	}

	void fill_cuBuffer(cuGaussianSplatting::cuGaussianBasic* basis_obj, int n_points, int n_offset)
	{
		// pos, rot, scale, opacity, shs
		cuGaussianSplatting::DeviceToDeviceWrapper(cuPoints.pos_cuda + 3 * n_offset, basis_obj->d_pos, n_points * 3 * sizeof(float));
		cuGaussianSplatting::RotActivation(basis_obj->d_rot, cuPoints.rot_cuda, n_points, n_offset);
		cuGaussianSplatting::ScaleActivation(basis_obj->d_scale, cuPoints.scale_cuda, n_points, n_offset);
		cuGaussianSplatting::OpacityActivation(basis_obj->d_opacity, cuPoints.opacity_cuda, n_points, n_offset);
		int n_elem_sh = basis_obj->get_elem_sh();
		cuGaussianSplatting::DeviceToDeviceWrapper(cuPoints.shs_cuda + n_elem_sh * n_offset, basis_obj->d_shs, n_points * n_elem_sh * sizeof(float));
	}

	void init() {
		
		//glClearColor(0.f, 0.f, 0.f, 0.0f);
		glClearColor(g_bkg_color[0], g_bkg_color[1], g_bkg_color[2], 0.0f);
		glShadeModel(GL_SMOOTH);
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_LIGHT0);

#ifdef ENABLE_MSAA
		glDisable(GL_MULTISAMPLE);
		//use_msaa = false;
		use_msaa = true;
#endif
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//std::string tmp_str;

		//// gaussian splatting
		{
			// options
			enable_deform_rot = true;
			enable_deform_rot_sh = true;
			enable_trans_rot = true;
			enable_trans_rot_sh = true;


			enable_mouth_rot = true;
			enable_mouth_rot_sh = true;

			// copy
			ui_deform_rot = enable_deform_rot; ui_deform_rot_sh = enable_deform_rot_sh;
			ui_trans_rot = enable_trans_rot; ui_trans_rot_sh = enable_trans_rot_sh;
			ui_mouth_rot = enable_mouth_rot; ui_mouth_rot_sh = enable_mouth_rot_sh;


			// load CPU objects
			//const char* filename = "./dataset/newexp4_nf_01_set6";
            const char * filename = g_model_path.c_str();

			//const char* iter_str = "40000";
			const char * iter_str = g_load_iter.c_str();			

			// set hair filename
			//tmp_str = "D:\\github\\HairStep\\debug\\biden\\dist3\\transfer_hair.obj";
			
			{
				std::string subname = std::string("fix_chkpnt") + iter_str;
				gs_face.load(gaussian_splatting::cat_path(filename, subname.c_str()).c_str());
			}
			{
				std::string subname = std::string("mouth0_chkpnt") + iter_str;
				gs_mouth0.load(gaussian_splatting::cat_path(filename, subname.c_str()).c_str());
			}
			{
				std::string subname = std::string("mouth1_chkpnt") + iter_str;
				gs_mouth1.load(gaussian_splatting::cat_path(filename, subname.c_str()).c_str());
			}
			//{
			//	std::string subname = std::string("hair_chkpnt") + iter_str;
			//	gs_hair.load(gaussian_splatting::cat_path(filename, subname.c_str()).c_str());
			//}
			
			const int n_points_face = gs_face.pos.shape[0];
			const int n_points_mouth0 = gs_mouth0.pos.shape[0];
			const int n_points_mouth1 = gs_mouth1.pos.shape[0];
			//const int n_points_hair = gs_hair.pos.shape[0];
			int i_offset = 0;
			g_face_offset = i_offset; i_offset += n_points_face;
			g_mouth0_offset = i_offset; i_offset += n_points_mouth0;
			g_mouth1_offset = i_offset; i_offset += n_points_mouth1;
			//g_hair_offset = i_offset; i_offset += n_points_hair;

			// transfer to GPU objects
			{
				cu_face.force_50 = true;
				cu_face.initParams(n_points_face);
				cu_face.setBasis(gs_face.getK());
				cu_face.AllocateData((bool)gs_face.rot_d.data<float>());
				cu_face.allocateNodeTransfer();
				cuGaussianSplatting::CheckCudaError(-1);
				cu_face.CopyData(
					gs_face.pos.data<float>(),
					gs_face.rot.data<float>(),
					gs_face.scale.data<float>(),
					gs_face.opacity.data<float>(),
					gs_face.shs.data<float>()
				);
				cuGaussianSplatting::CheckCudaError(-2);
				cu_face.CopyDataEx(
					gs_face.xyz_t.data<float>(),
					gs_face.rot_t.data<float>(),
					gs_face.scale_t.data<float>(),
					gs_face.opacity_t.data<float>(),
					gs_face.shs_t.data<float>(),
					gs_face.rot_d.data<float>(),
					gs_face.pos_t.data<float>(),
					gs_face.W.data<float>(),
					gs_face.eyelid.data<float>()
				);
				cuGaussianSplatting::CheckCudaError(-3);
			}
			{
				cu_mouth0.initParams(n_points_mouth0);
				cu_mouth0.AllocateData();
				cu_mouth0.allocateTransferData();
				cu_mouth0.CopyData(
					gs_mouth0.pos.data<float>(),
					gs_mouth0.rot.data<float>(),
					gs_mouth0.scale.data<float>(),
					gs_mouth0.opacity.data<float>(),
					gs_mouth0.shs.data<float>()
				);
			}
			{
				cu_mouth1.initParams(n_points_mouth1);
				cu_mouth1.AllocateData();
				cu_mouth1.allocateTransferData();
				cu_mouth1.CopyData(
					gs_mouth1.pos.data<float>(),
					gs_mouth1.rot.data<float>(),
					gs_mouth1.scale.data<float>(),
					gs_mouth1.opacity.data<float>(),
					gs_mouth1.shs.data<float>()
				);
			}
			{
				bool flag = gb_shader.LoadFromFile("pc_color",
                                                   "shader/pc_color.vp",
                                                   "shader/pc_color_scale.fp",
                                                   2, 0, "point", 1, "color");
				if (!flag) throw std::runtime_error("shader load fail pc_color");
			}
			{
				// Use SSBO
				bool flag = gb_shader.LoadFromFile("gs_copy",
                                                   "shader/copy.vp",
                                                   "shader/copy.fp",
                                                   1, 0, "in_vertex");
				if (!flag) throw std::runtime_error("shader load fail gs_copy");
			}

			cuTrans.Initialize();
			
			// currently we use hair's point 
			int n_total_points = i_offset;

			cuPoints.initParams(n_total_points);
			cuPoints.allocateData(true,true);// set color -> true
			cuPoints.allocateRectData();
			cuPoints.allocateBkgData(false,g_bkg_color);

			cuGaussianSplatting::CheckCudaError(0);			

			fill_cuBuffer(&cu_face, n_points_face, g_face_offset);
			
			cuGaussianSplatting::CheckCudaError(78);

			fill_cuBuffer(&cu_mouth0, n_points_mouth0, g_mouth0_offset);
			fill_cuBuffer(&cu_mouth1, n_points_mouth1, g_mouth1_offset);

			geomBufferFunc = cuGaussianSplatting::resizeFunctional(&geomPtr, allocdGeom);
			binningBufferFunc = cuGaussianSplatting::resizeFunctional(&binningPtr, allocdBinning);
			imgBufferFunc = cuGaussianSplatting::resizeFunctional(&imgPtr, allocdImg);

			// Gaussian Splatting use SSBO

			cuBuffer = new gaussian_splatting::gl_buffer_object();
			cuBuffer->Init();
			//cuBuffer->Allocate(sizeof(float) * 3 * winHeight * winWidth, nullptr, GL_DYNAMIC_STORAGE_BIT);
			cuBuffer->Allocate(sizeof(float) * 3 * winHeight * winWidth, nullptr, GL_DYNAMIC_COPY);
			// register to slot 1
			cuGaussianSplatting::RegisterGLBuffers(cuBuffer->bufferID, cuGaussianSplatting::WRITEDISCARD, 1);
		}
		//// FLAME
		{
			const char* mesh_file = "./data/head_template_mesh.obj";
			g_face_mesh.loadObj(mesh_file, true);
			if (true) {
				float offset_[3] = { 0.f,-1.5f,0.f };
				g_face_mesh.offset(offset_);
				//g_face_mesh.scale(g_face_mesh_scale);
			}


			load_back_head_index("./data/FLAME2020/back_of_head.txt");

			const char * FLAME_dir = "./data/FLAME2020";
			g_flame.load(FLAME_dir);

			faceParams.FindDiff();
			should_update_mesh = true;
			should_update_gs = true;

			cuParams.Initialize(512 * sizeof(float));
		}


		loadFrameSeq();
	}


	void destroy() {

		cu_face.Destroy();
		cu_mouth0.Destroy();
		cu_mouth1.Destroy();

		gb_shader.Clear();

		{
			cuPoints.Destroy();
			cuTrans.Destroy();
			if (cuBuffer) {
				// register from slot 1
				cuGaussianSplatting::UnregisterGLResources(1);
			}
			delete cuBuffer; cuBuffer = nullptr;
		}
		torch_frames.clear();
		if (camera_vis) delete camera_vis;
	}

	void framebuffer_size_callback(GLFWwindow* window, int w, int h)
	{
		winWidth = w;
		winHeight = h;
		glViewport(0, 0, w, h);

		glMatrixMode(GL_PROJECTION);//
		glLoadIdentity();
		gluPerspective(g_fovy, (float)w / (float)h, 0.01f, 20.f);
		glMatrixMode(GL_MODELVIEW);

		// minimized window may give you (0,0,0,0) viewport, which is problematic
		if (winWidth == 0 || winHeight == 0)
			return; 

		if (cuBuffer) {
			cuGaussianSplatting::UnregisterGLResources(1); // re-register is required
			//cuBuffer->Allocate(sizeof(float) * 3 * winHeight * winWidth, nullptr, GL_DYNAMIC_STORAGE_BIT);
			cuBuffer->Allocate(sizeof(float) * 3 * winHeight * winWidth, nullptr, GL_DYNAMIC_COPY);
			cuGaussianSplatting::RegisterGLBuffers(cuBuffer->bufferID, cuGaussianSplatting::WRITEDISCARD, 1);
		}
	}

	void cursor_position_callback(GLFWwindow* window, double x, double y)
	{
		if (ImGui::GetIO().WantCaptureMouse) return;

		//printf("Mouse position move to [%f:%f]\n", x, y);
		if (left_down) {
			rotter.leftbutton_trace(winWidth, winHeight, x, y);
		}
		else if (right_down) {
			float ratio = tan(g_fovy * M_PI / 360.f) / (winHeight * 0.5f);
			rotter.rightbutton_trace(-ratio * zoom, -ratio * zoom, x, y);
		}
	}

	void scroll_callback(GLFWwindow* window, double x, double y)
	{
		if (ImGui::GetIO().WantCaptureMouse) return;
		zoom += y * 0.05f;
		//zoom += y * 0.3f;
	}


	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
	{
		//ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
		if (ImGui::GetIO().WantCaptureMouse) return;

		double x, y;
		glfwGetCursorPos(window, &x, &y);
		int xpos = int(x);
		int ypos = int(y);

		glutDoubleClick::MouseOperation op, lastop;
		op.button = button;
		op.state = action;
		op.x = xpos;
		op.y = ypos;
		bool flag = gdc.setMouseOperation(op, lastop);

		if (action == GLFW_PRESS) {
			if (button == GLFW_MOUSE_BUTTON_LEFT)
			{
				if (flag && lastop.button == GLFW_MOUSE_BUTTON_LEFT && lastop.state == GLFW_RELEASE) {
					// double click
					rotter.reset();
					zoom = zoom_init;
				}
				else {
					left_down = true;
					rotter.leftbutton_down(winWidth, winHeight, xpos, ypos);
				}
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT)
			{
				rotter.rightbutton_down(xpos, ypos);
				right_down = true;
			}
		}
		else if (action == GLFW_RELEASE) {
			if (button == GLFW_MOUSE_BUTTON_LEFT) {
				if (left_down) { // for double click bug. 
					left_down = false;
					rotter.leftbutton_up(winWidth, winHeight, xpos, ypos);
				}
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
				right_down = false;
				rotter.rightbutton_down(xpos, ypos);
			}
		}
		return;
	}

	void RenderGaussianSplatting() {
		
		glDisable(GL_DEPTH_TEST);

		//gaussian_splatting::SetGLTransforms(cuTrans);
		gaussian_splatting::SetGLTransforms2(cuTrans);
		if (cuTrans.vpParams[2] == 0 || cuTrans.vpParams[3] == 0) {
			return;// this may happen when window is minimized ...
		}

		//const int P = pc_obj.pc.n_vertex;
		//const int P = cuPoints.n_points;
		const int D = 3;
		const int M = 16;
		// TODO, make me global later
		bool gsp_cropping = false;
		bool gsp_fastCulling = true;
		int* rects = gsp_fastCulling ? cuPoints.rect_cuda : nullptr;
		//float* boxmin = gsp_cropping ? (float*)&_boxmin : nullptr;
		//float* boxmax = gsp_cropping ? (float*)&_boxmax : nullptr;
		float* boxmin = nullptr;
		float* boxmax = nullptr;

		float aspect = float(winWidth) / float(winHeight);
		float rad_fovy = g_fovy * M_PI / 180.f;
		float tan_fovy = tan(rad_fovy * 0.5f);
		float tan_fovx = tan_fovy * aspect;

		float* image_cuda;
		size_t bytes;

		// bind slot 1
		cuGaussianSplatting::GLMappedPointer((void**)&image_cuda, bytes, 1);

		const float* ptr_shs = nullptr;
		const float* ptr_colors_precomp = nullptr;		
		if (render_mode == 1) { // render isotropic point
			ptr_shs = cuPoints.shs_cuda; 
		}
		else if (render_mode == 2) {// std mode
			ptr_shs = cuPoints.shs_cuda;
			//printf("%x %x\n", ptr_shs, ptr_colors_precomp);
		}
		else if (render_mode == 3) { // vis opacity
			float dummy_one[3] = { 1,1,1 };
			cuPoints.fillColor(dummy_one);
			ptr_colors_precomp = cuPoints.color_cuda;
		}
		else if (render_mode == 4) { // vis anisotropy
			cuPoints.fillColorWithAnisotropy(max_show_anistropy);
			ptr_colors_precomp = cuPoints.color_cuda;		
		}
		else if (render_mode == 5) { // vis depth
			cuPoints.fillColorWithDepth((float *)cuTrans.matrices);
			ptr_colors_precomp = cuPoints.color_cuda;		
		}
		
		int Offset = 0;
		int P = 0;
		if (render_part_code == 0) { // all
			Offset = 0;
			P = cuPoints.n_points;
		}
		else if (render_part_code == 1) { // face
			Offset = g_face_offset;
			P = cu_face.n_points;		
		}
		else if (render_part_code == 2) { // mouth0
			Offset = g_mouth0_offset;
			P = cu_mouth0.n_points;
		}
		else if (render_part_code == 3) { // mouth1
			Offset = g_mouth1_offset;
			P = cu_mouth1.n_points;
		}
		else if (render_part_code == 4) { // mouth0 + mouth1
			Offset = g_mouth0_offset;
			P = cu_mouth0.n_points + cu_mouth1.n_points;
		}

		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			P, D, M,
			cuPoints.bkg_cuda,
			winWidth, winHeight,
			cuPoints.pos_cuda + 3 * Offset,
			ptr_shs + 3 * 16 * Offset,
			ptr_colors_precomp == nullptr ? nullptr :  ptr_colors_precomp + 3 * Offset, // colors_precomp
			cuPoints.opacity_cuda + 1 * Offset,
			render_mode == 1 ? nullptr : cuPoints.scale_cuda + 3 * Offset,
			gsp_scalingModifer,
			cuPoints.rot_cuda + 4 * Offset,
			nullptr, // cov3d_precomp
			(float*)cuTrans.matrices + 16, // mvMatrix
			//(float *)cuTrans.matrices + 48, // projMatrix
			(float*)cuTrans.matrices, // mvpMatrix
			(float*)cuTrans.matrices + 32 + 12, // campos, part of c2wMatrix
			tan_fovx, 
			tan_fovy, 
			false, // prefiltered
			image_cuda, // output color
			nullptr, // radii
			rects, // rects
			boxmin,  
			boxmax  
		);
		
		// gray --> hotmap
		if (render_mode == 5) {
			float modelview[16];
			glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat*)modelview);
			float z = -modelview[14];
			//printf("z %f\n", z);
			float k, b;			
			// y = 0.5*(s*(x-z)) + 0.5
			k = 0.5f * scale_show_depth;
			b = 0.5f - 0.5f * scale_show_depth * z;			
			cuPoints.ColorDepth(image_cuda, winWidth, winHeight, k, b, true);
		}

		cuGaussianSplatting::GLUnmapPointer(1); // 

		// SSBO solution
		gb_shader.SetCurrentShader("gs_copy");
		gb_shader.UseCurrentShader();
		//int _flip = false;
		int _flip = true;
		gb_shader.SetUniformParams("flip", ShaderManager::VEC1I, 1, &_flip);
		gb_shader.SetUniformParams("width", ShaderManager::VEC1I, 1, &winWidth);
		gb_shader.SetUniformParams("height", ShaderManager::VEC1I, 1, &winHeight);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cuBuffer->bufferID);
		glBegin(GL_QUADS);
		glVertex3f(-1.f, -1.f, 0.f);
		glVertex3f(1.f, -1.f, 0.f);
		glVertex3f(1.f, 1.f, 0.f);
		glVertex3f(-1.f, 1.f, 0.f);
		glEnd();
		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
		gb_shader.UseDefaultShader();

		glEnable(GL_DEPTH_TEST);
	
	}

	void update_face();
	void update_mouth();

	void draw_pc() {
		
		if (should_update_gs) {

			//printf("update mesh called!\n");

			auto start = std::chrono::steady_clock::now();
			// do something

			const std::vector<int>& sel_id = loc_transfer.back_head_vertex;
			std::vector<Eigen::Vector3f> backhead_vertex_expr(sel_id.size());
			std::vector<Eigen::Vector3f> backhead_vertex_zero(sel_id.size());


			gaussian_splatting::lbs_acc(
				backhead_vertex_expr,
				sel_id, &g_flame,
				faceParams.params_shape, faceParams.n_params_shape,
				faceParams.params_expr, faceParams.n_params_expr,
				faceParams.params_pose,
				faceParams.params_eyelid
			);

			// copy from tmp buffer
			float* A_ptr = loc_transfer.A;
			for (int i = 0; i < 5 * 4 * 4; i++) {
				A_ptr[i] = gaussian_splatting::debug_outA[i];
			}

			auto finish = std::chrono::steady_clock::now();
			double elapsed_seconds = std::chrono::duration_cast<
				std::chrono::duration<double>>(finish - start).count();
			record_time1 += elapsed_seconds;


			{
				gaussian_splatting::lbs_acc(
					backhead_vertex_zero,
					sel_id, &g_flame,
					faceParams.params_shape, faceParams.n_params_shape,
					zeroFaceParams.params_expr, zeroFaceParams.n_params_expr,
					zeroFaceParams.params_pose,
					zeroFaceParams.params_eyelid
				);

				// copy from tmp buffer
				float* A0_ptr = loc_transfer.A0;
				for (int i = 0; i < 5 * 4 * 4; i++) {
					A0_ptr[i] = gaussian_splatting::debug_outA[i];
				}
				//gaussian_splatting::debug_get_jaw_transform(jawTrans0);
			}
			// compute rjawTrans & rjawTransN
			loc_transfer.ComputeJawTransFromA();
			// compute frameR,t & frameNR
			loc_transfer.ComputeBackHeadVertexDirect(
				backhead_vertex_expr,
				backhead_vertex_zero
			);

			update_face();
			//update_hair();
			update_mouth();			

			should_update_gs = false;
		}

		if (render_mode >= 1) {
			RenderGaussianSplatting();
		}
	
	}

	void draw_obj() {		

		if (should_update_mesh) {

			//printf("update mesh called!\n");

			auto start = std::chrono::steady_clock::now();
			
			gaussian_splatting::lbs(
				g_face_mesh.vertex_list,
				&g_flame,
				faceParams.params_shape, faceParams.n_params_shape,
				faceParams.params_expr, faceParams.n_params_expr,
				faceParams.params_pose,
				faceParams.params_eyelid
				//g_face_mesh_scale
			);
			// copy from tmp buffer
			float* A_ptr = loc_transfer.A;
			for (int i = 0; i < 5 * 4 * 4; i++) {
				A_ptr[i] = gaussian_splatting::debug_outA[i];
			}
			//gaussian_splatting::debug_get_jaw_transform(jawTrans);

			//auto finish = std::chrono::steady_clock::now();
			//double elapsed_seconds = std::chrono::duration_cast<
			//	std::chrono::duration<double>>(finish - start).count();
			//record_time1 += elapsed_seconds;

			g_face_mesh.afterPoseChanged();

			should_update_mesh = false;

			auto finish = std::chrono::steady_clock::now();
			double elapsed_seconds = std::chrono::duration_cast<
				std::chrono::duration<double>>(finish - start).count();
			record_time1 += elapsed_seconds;

		}

		int mesh_ren_mode = 0;
		if (show_face) {
			mesh_ren_mode |= (SMesh::SW_LIGHTING | SMesh::SW_F);
			if (is_face_smooth) {
				mesh_ren_mode |= SMesh::SW_SMOOTH;
			}
			else {
				mesh_ren_mode |= SMesh::SW_FLAT;
			}
		}
		if (show_edge) {
			mesh_ren_mode |= SMesh::SW_E;
		}
		//if (use_texture) {
		//	mesh_ren_mode |= SMesh::SW_TEXTURE;
		//}
		
		g_face_mesh.render(mesh_ren_mode);
		//g_face_mesh.render(SMesh::SW_LIGHTING | SMesh::SW_F | SMesh::SW_SMOOTH);		

		// debug drawing ...
		if (show_edge) {

			// show back head vertex
			glColor3f(0.f, 1.f, 0.f);
			glPointSize(3.f);
			glBegin(GL_POINTS);
			const std::vector<int>& back_head_vertex = loc_transfer.back_head_vertex;
			for (int i = 0; i < back_head_vertex.size(); i++) {
				glVertex3fv(g_face_mesh.vertex_list[back_head_vertex[i]].data());
			}
			glEnd();
			glPointSize(1.f);		
		}

	}

	void draw_axis() {

		glBegin(GL_LINES);
		float line_size = 1.f;
		glColor3f(1.f, 0.0f, 0.0f);//x
		glVertex3f(0.0f, 0.f, 0.f);
		glVertex3f(line_size, 0.f, 0.f); 

		glColor3f(0.0f, 1.f, 0.0f);//y
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, line_size, 0.f); 

		glColor3f(0.0f, 0.0f, 1.f);//z
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, line_size);
		glEnd();
	}

	void copy_params() {

		//// prepare params
		// 100 expr + 2 eyelid + 36 pos
		float params_buffer[256];
		for (int i = 0; i < faceParams.n_params_expr; i++) {
			params_buffer[i] = faceParams.params_expr[i];
		}
		for (int i = 0; i < 2; i++) {
			params_buffer[100 + i] = faceParams.params_eyelid[i];
		}
		const int n_node = 5;
		float rot_mats[n_node * 3 * 3];
		gaussian_splatting::exp_so3(rot_mats, faceParams.params_pose, n_node);
		// remove identity
		for (int i = 0; i < n_node; i++) {
			rot_mats[i * 9 + 0] -= 1.f;
			rot_mats[i * 9 + 4] -= 1.f;
			rot_mats[i * 9 + 8] -= 1.f;
		}
		// copy 5 node except the first one
		for (int i = 1; i < n_node; i++) {
			for (int j = 0; j < 9; j++) {
				params_buffer[102 + (i - 1) * 9 + j] = rot_mats[i * 9 + j];
			}
		}
		cuParams.Copy(params_buffer, sizeof(float) * (100 + 2 + 36));
		
		//cuGaussianSplatting::CheckCudaError(99);

		//// compute A
		//const int n_node = 5;
		float A_buffer[n_node * 4 * 4];
		loc_transfer.ComputeRelA(A_buffer);
		cu_face.copyNodeTransfer(A_buffer);

		//cuGaussianSplatting::CheckCudaError(100);
	}

	void update_face() {
		
		copy_params();

		int n_offset = g_face_offset;
		

		cuGaussianSplatting::CompositeNewPipev1(
			(float*)cuParams.params,
			(float*)cu_face.d_node_transfer,
			&cu_face, &cuPoints, gs_face.getK(),
			enable_deform_rot, enable_deform_rot_sh,
			enable_trans_rot, enable_trans_rot_sh,
			n_offset, true
		);
	}

	//void update_hair() {		
	//	//printf("update hair called!\n");

	//	int n_offset = g_hair_offset;
	//	int n_points = cu_hair.n_points;
	//	
	//	// this two are not affected, may even not copy it
	//	//cuGaussianSplatting::ScaleActivation(cu_hair.d_scale, cuPoints.scale_cuda, n_points, n_offset);
	//	//cuGaussianSplatting::OpacityActivation(cu_hair.d_opacity, cuPoints.opacity_cuda, n_points, n_offset);

	//	const float* frameR = loc_transfer.frameR;
	//	const float* frameNR = loc_transfer.frameNR;

	//	//for (int i = 0; i < 4; i++) {
	//	//	for (int j = 0; j < 4; j++) {
	//	//		printf("%.4f  ", frameR[4 * i + j]);
	//	//	}printf("\n");
	//	//}

	//	cu_hair.copyTransferData(frameR, frameNR);
	//	// affect pos, rot, sh
	//	cuGaussianSplatting::SimpleTransferBasic(
	//		&cu_hair, &cuPoints, enable_hair_rot, enable_hair_rot_sh, n_offset 
	//	);	
	//}

	

	void update_mouth() {
		
		//printf("update mouth called!\n");
		{
			int n_offset = g_mouth0_offset;
			int n_points = cu_mouth0.n_points;
			const float* frameR = loc_transfer.frameR;
			const float* frameNR = loc_transfer.frameNR;
			cu_mouth0.copyTransferData(frameR, frameNR);
			cuGaussianSplatting::SimpleTransferBasic(
				&cu_mouth0, &cuPoints, enable_mouth_rot, enable_mouth_rot_sh, n_offset
			);
		}
		{
			int n_offset = g_mouth1_offset;
			int n_points = cu_mouth1.n_points;
			const float* rjawTrans = loc_transfer.rjawTrans;
			const float* rjawTransN = loc_transfer.rjawTransN;
			cu_mouth1.copyTransferData(rjawTrans, rjawTransN);
			cuGaussianSplatting::SimpleTransferBasic(
				&cu_mouth1, &cuPoints, enable_mouth_rot, enable_mouth_rot_sh, n_offset
			);			
		}
	}

	void draw_camera(){
		
		float p_size;
		
		glGetFloatv(GL_POINT_SIZE, &p_size);
		glPointSize(3.f);
		int n_seg = camera_vis->size() - n_test; 
		if (n_seg < 0) n_seg = 0;
		glColor3f(1.f, 1.f, 0.f);
		draw_camera_pos(*camera_vis, 0, 0, n_seg);
		glColor3f(0.f, 1.f, 1.f);
		draw_camera_pos(*camera_vis, 0, n_seg, camera_vis->size() - n_seg);
		glPointSize(p_size);

		glColor3f(0.8f, 0.8f, 0.8f);
		glGetFloatv(GL_LINE_WIDTH, &p_size);
		draw_camera_pos(*camera_vis, 1);
		glLineWidth(p_size);
		draw_camera_frame(*camera_vis, sel_camera);

	}

	void loadFrameCameras(float* c) {

		gaussian_splatting::TorchFrame& frame = torch_frames[sel_camera];
		std::vector<float>& R = frame.params["R"];
		std::vector<float>& t = frame.params["t"];
		float w2c[16];
		w2c[0] = R[0]; w2c[4] = R[1]; w2c[8] = R[2]; w2c[12] = t[0];
		w2c[1] = -R[3]; w2c[5] = -R[4]; w2c[9] = -R[5]; w2c[13] = -t[1]; // flip Y in camera coord
		w2c[2] = -R[6]; w2c[6] = -R[7]; w2c[10] = -R[8]; w2c[14] = -t[2]; // flip Z in camera coord
		w2c[3] = 0; w2c[7] = 0; w2c[11] = 0; w2c[15] = 1;
		if (true) { // NOTE: Move the object on screen slightly to the right 
			w2c[12] += 0.05f;
		}
		for (int i = 0; i < 16; i++) {
			c[i] = w2c[i];
		}
	}

	void display() {
		
		// minimized window may give you (0,0,0,0) viewport, which is wired
		if (winWidth == 0 || winHeight == 0)
			return;

		if (record_flag) {
			record_n += 1;
		}

		if (enable_rot) {
			rotter.idleRot(gti_rotate.touch() * 1e-4f, 1);
		}
		if (enable_play) { 
			float ms = 1000.f / (float)play_framerate;
			if (gti_video.compareThreshold(int(round(ms)), true)) {
				int n_frames = torch_frames.size();
				int test_start = n_frames - n_test;
				if (sel_camera >= test_start) {
					sel_camera = sel_camera + 1;
					if (sel_camera >= n_frames) {
						sel_camera = test_start;
					}
				}
				else {
					sel_camera = sel_camera + 1;
					if (sel_camera >= test_start) {
						sel_camera = 0;
					}
				}
				should_update_frame = true;
			}
		}

		// when MSAA is enabled, points seem to be rendered as spheres ...
#ifdef ENABLE_MSAA
		if (use_msaa) {
			glEnable(GL_MULTISAMPLE);
		}
		else {
			glDisable(GL_MULTISAMPLE);
		}
#endif

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		float t[3];	float c[16];
		if ((enable_play && enable_play_camera) ||
			(!enable_play && enable_fix_camera)) {
			loadFrameCameras(c);
			glMultMatrixf(c);
		}
		else {
			rotter.getTrans(t);
			glTranslatef(t[0], t[1], zoom);

			rotter.getRot(c);
			glMultMatrixf(c);
		}

		// Update global state
		if (should_update_bkg) {
			glClearColor(g_bkg_color[0], g_bkg_color[1], g_bkg_color[2],0.0);
			cuPoints.allocateBkgData(false, g_bkg_color);
			should_update_bkg = false;
		}

		if (should_update_frame) {
			loadInfoFromFrame(&torch_frames[sel_camera]);
			should_update_frame = false;
		}


		if (display_obj) {
			if (render_mode == 0)
				draw_obj();
			else 
				draw_pc();
		}
			

		if (display_axis)
			draw_axis();
		if (camera_vis && display_cam)
			draw_camera();

		glPopMatrix();

	}

	void loadInfoFromFrame(gaussian_splatting::TorchFrame* frame) {

		if (frame == nullptr) return;
		//printf("load info from frame\n");
		// copy identity, expr to UI

		// [UGLY BUG FIX] twice update
		// this function always set should_update_mesh -> True, and it triggers update immediately
		// However, it may also lead to inconsistency of params and ui_params, which will set should_update_mesh -> True later in drawGui()
		// and another update will be triggered.

		//std::vector<float>& tex = frame->params["tex"];
		//for (size_t i = 0; i < tex.size(); i++) {
		//	faceTexParams.params_tex[i] = tex[i];
		//	// HACKY, this avoid twice update 
		//	faceTexParams.ui_params_tex[i] = tex[i];
		//}
		std::vector<float>& shape = frame->params["shape"];
		for (size_t i = 0; i < shape.size(); i++) {
			faceParams.params_shape[i] = shape[i];
			// HACKY, this avoid twice update 
			faceParams.ui_params_shape[i] = shape[i];
		}
		std::vector<float>& exp = frame->params["exp"];
		for (size_t i = 0; i < exp.size(); i++) {
			faceParams.params_expr[i] = exp[i];
			// HACKY, this avoid twice update 
			faceParams.ui_params_expr[i] = exp[i];
		}
		// load camera (approximate)
		static int loc_flag = 0; // load only once

		if(loc_flag == 0)
		{
			std::vector<float>& R = frame->params["R"];
			std::vector<float>& t = frame->params["t"];
			float w2c[16],c2w[16];
			w2c[0] = R[0]; w2c[4] = R[1]; w2c[8] = R[2]; w2c[12] = t[0];
			w2c[1] = R[3]; w2c[5] = R[4]; w2c[9] = R[5]; w2c[13] = t[1];
			w2c[2] = R[6]; w2c[6] = R[7]; w2c[10] = R[8]; w2c[14] = t[2];
			w2c[3] = 0; w2c[7] = 0; w2c[11] = 0; w2c[15] = 1;
			MatrixTools::inv_of_unity16(w2c, c2w);
			//printf("%f %f %f\n", c2w[12], c2w[13], c2w[14]);
			zoom = zoom_init = -sqrt(c2w[12] * c2w[12] + c2w[13] * c2w[13] + c2w[14] * c2w[14]);
		}
		

		if(loc_flag == 0) 
		{
			std::vector<float>& K = frame->params["K"];
			//printf("Matrix K\n");
			//for (int i = 0; i < 3; i++) {
			//	for (int j = 0; j < 3; j++) {
			//		printf("%f ", K[3 * i + j]);
			//	}printf("\n");
			//}
			float dataset_H = 512; // TODO, make me flexible
			float tmp = dataset_H / (2 * K[4]);
			ui_fovy = g_fovy = atan(tmp) * 2 * (180. / M_PI);
			//printf("load fovy %f ,suppose dataset height is %f\n", g_fovy, dataset_H);

			glMatrixMode(GL_PROJECTION);//
			glLoadIdentity();
			gluPerspective(g_fovy, (float)winWidth / (float)winHeight, 0.01f, 20.f);
			glMatrixMode(GL_MODELVIEW);
		}

		loc_flag = 1;

		// jaw, eyes, eyelids
		{
			std::vector<float>& jaw = frame->params["jaw"];
			float tmp[9];
			gaussian_splatting::rotation_6d_to_matrix(tmp, jaw.data(), 1);
			gaussian_splatting::log_so3(&faceParams.params_pose[6], tmp,1);//bug ?
			// HACKY, this avoid update 
			for (int i = 6; i < 9; i++) {
				faceParams.ui_params_pose[i] = faceParams.params_pose[i];
			}
		}
		{
			std::vector<float>& eyes = frame->params["eyes"];
			float tmp[9 * 2];
			gaussian_splatting::rotation_6d_to_matrix(tmp, eyes.data(), 2);
			gaussian_splatting::log_so3(&faceParams.params_pose[9], tmp,2);
			// HACKY, this avoid update 
			for (int i = 9; i < 9+6; i++) {
				faceParams.ui_params_pose[i] = faceParams.params_pose[i];
			}
		}
		// load eyelids
		std::vector<float>& eyelids = frame->params["eyelids"];
		for (size_t i = 0; i < eyelids.size(); i++) {
			faceParams.params_eyelid[i] = eyelids[i];
			// HACKY, this avoid update 
			faceParams.ui_params_eyelid[i] = eyelids[i];
		}

		should_update_mesh = true;
		should_update_gs = true;

	}


	void loadFrames(const char* filename) {

		//printf("load frame %s\n", filename);
		char buff[512];
		int n = 0;
		while (true) {
			sprintf(buff, "%05d.bin", n);
			std::string name = gaussian_splatting::cat_path(filename, buff);
			if (!gaussian_splatting::isValidFile(name.c_str()))
				break;
			n = n + 1;
			if (n % 100 == 0) {
				printf("%d\n", n);
			}
		}
		if (n == 0)
			throw std::runtime_error("Detect 0 frames!");
		printf("find %d files in folder %s\n", n, filename);
		torch_frames.resize(n);
		for (int ii = 0; ii < n; ii++) {
			sprintf(buff, "%05d.bin", ii);
			std::string name = gaussian_splatting::cat_path(filename, buff);
			torch_frames[ii].load(name.c_str());
			//if (ii % 100 == 99) {
			//	printf("[%d/%d]\n", ii, n);
			//}
		}
		//
		should_update_frame = true;
	}

	void loadCameraTraj() {
		if (true) {
			if (camera_vis == nullptr) {
				camera_vis = new gaussian_splatting::CameraVisualizer();
				camera_vis->load(torch_frames.data(), torch_frames.size());
				camera_vis->update(0.1f);
			}
		}
	}

	void loadFrameSeq() {

		//const char* params_dir = "./dataset/params";
		const char* params_dir = g_dataset_path.c_str();

		loadFrames(params_dir);
		loadCameraTraj();
	}

	void drawGui() {
		
		int child_seq = 0;

		bool should_check_diff = false;
		//should_update_mesh = false;
		//should_update_gs = false;
		char cbuff[256];
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		ImGui::SetNextWindowBgAlpha(0.3f);
		//ImGui::SetNextWindowBgAlpha(0.0f);

		ImGui::Begin("Render Setting", nullptr, ImGuiWindowFlags_MenuBar);

		if (ImGui::CollapsingHeader("Frames")) {
			char buff[512];
			int n_frames = torch_frames.size();
			int test_start = n_frames - n_test; 
			sprintf(buff,"%d/%d, test set start from %d\n",sel_camera, n_frames, test_start );
			ImGui::Text(buff);
			ImGui::SliderInt("frame", &sel_camera, 0, n_frames - 1);
			ImGui::SliderInt("play fps", &play_framerate, 1, 30);

			ImGui::Checkbox("camera", &enable_play_camera);
			ImGui::Checkbox("fix camera", &enable_fix_camera);

			if (sel_camera != ui_sel_camera) {
				ui_sel_camera = sel_camera;
				loadInfoFromFrame(&torch_frames[sel_camera]);
			}
			if (ImGui::Button("Head")) {
				sel_camera = 0; ui_sel_camera = sel_camera;
				loadInfoFromFrame(&torch_frames[sel_camera]);
			}ImGui::SameLine();
			if (ImGui::Button("Test")) {
				sel_camera = test_start; ui_sel_camera = sel_camera;
				loadInfoFromFrame(&torch_frames[sel_camera]);
			}ImGui::SameLine();
			if (ImGui::Button("End")) {
				sel_camera = n_frames - 1; ui_sel_camera = sel_camera;
				loadInfoFromFrame(&torch_frames[sel_camera]);
			}
		}

		if (ImGui::CollapsingHeader("Settings")) {

			//ImGui::Text("Render Mode");
			sprintf(cbuff, "Render mode %d\n", render_mode + 1);
			ImGui::Text(cbuff);

			sprintf(cbuff, "Show obj (Key O) %s\n", display_obj ? "True" : "False");
			ImGui::Text(cbuff);
			sprintf(cbuff, "Show axis (Key C) %s\n", display_axis ? "True" : "False");
			ImGui::Text(cbuff);
			sprintf(cbuff, "Show camera (Key T) %s\n", display_cam ? "True" : "False");
			ImGui::Text(cbuff);
			sprintf(cbuff, "Render smooth (Key S) %s\n", is_face_smooth ? "True" : "False");
			ImGui::Text(cbuff);
			sprintf(cbuff, "Show surface (Key F) %s\n", show_face ? "True" : "False");
			ImGui::Text(cbuff);
			sprintf(cbuff, "Show edge (Key E) %s\n", show_edge ? "True" : "False");
			ImGui::Text(cbuff);

			if (ImGui::Button("Force Update Mesh")) {
				should_update_mesh = true;
				should_update_gs = true;
			}ImGui::SameLine();
			if (ImGui::Button("Reload Frame")) {
				//loadInfoFromFrame(torch_frame);
				loadInfoFromFrame(torch_frames.data() + sel_camera);
			}ImGui::SameLine();
			if (ImGui::Button("See Expr Basis")) {
				faceParams.resetExprParams();
				faceParams.resetPoseParams();
				faceParams.resetEyelidParams();
				should_update_mesh = true;
				should_update_gs = true;
			}

			ImGui::Text("Part");
			ImGui::RadioButton("All", &render_part_code, 0); ImGui::SameLine();
			ImGui::RadioButton("Face", &render_part_code,1);
			ImGui::RadioButton("Mouth0", &render_part_code, 2); ImGui::SameLine();
			ImGui::RadioButton("Mouth1", &render_part_code, 3);
			ImGui::RadioButton("Mouth", &render_part_code, 4); //ImGui::SameLine();

			ImGui::Text("Flag");
			ImGui::Checkbox("deform rot", &enable_deform_rot); ImGui::SameLine();
			ImGui::Checkbox("deform rot sh", &enable_deform_rot_sh);
			ImGui::Checkbox("trans rot", &enable_trans_rot); ImGui::SameLine();
			ImGui::Checkbox("trans rot sh", &enable_trans_rot_sh); 
			ImGui::Checkbox("mouth rot", &enable_mouth_rot); ImGui::SameLine();
			ImGui::Checkbox("mouth rot sh", &enable_mouth_rot_sh);
			if (ui_deform_rot != enable_deform_rot) { ui_deform_rot = enable_deform_rot; should_update_mesh = true; should_update_gs = true; }
			if (ui_deform_rot_sh != enable_deform_rot_sh) { ui_deform_rot_sh = enable_deform_rot_sh; should_update_mesh = true; should_update_gs = true;}
			if (ui_trans_rot != enable_trans_rot) { ui_trans_rot = enable_trans_rot; should_update_mesh = true; should_update_gs = true;}
			if (ui_trans_rot_sh != enable_trans_rot_sh) { ui_trans_rot_sh = enable_trans_rot_sh; should_update_mesh = true; should_update_gs = true;}
			if (ui_mouth_rot != enable_mouth_rot) { ui_mouth_rot = enable_mouth_rot; should_update_mesh = true; should_update_gs = true;}
			if (ui_mouth_rot_sh != enable_mouth_rot_sh) { ui_mouth_rot_sh = enable_mouth_rot_sh; should_update_mesh = true; should_update_gs = true;}

			ImGui::SliderFloat("PointSize", &point_size, 1.f, 8.f);
			ImGui::SliderFloat("PointAtt", &point_color_att, 0.1f, 1.f);
			ImGui::SliderFloat("Scaling Modifier", &gsp_scalingModifer, 0.001f, 1.2f);
			if (ImGui::Button("Reset")) {
				gsp_scalingModifer = 1.f;
			}
			ImGui::SliderFloat3("Bkg Color", g_bkg_color, 0.0, 1.0);
			for (int i = 0; i < 3; i++) {
				if (g_bkg_color[i] != ui_bkg_color[i]) {
					ui_bkg_color[i] = g_bkg_color[i];
					should_update_bkg = true;
				}
			}
			if (ImGui::Button("Black")) {
				ui_bkg_color[0] = g_bkg_color[0] = 0;
				ui_bkg_color[1] = g_bkg_color[1] = 0;
				ui_bkg_color[2] = g_bkg_color[2] = 0;
				should_update_bkg = true;
			}ImGui::SameLine();
			if (ImGui::Button("Green")) {
				ui_bkg_color[0] = g_bkg_color[0] = 0.6f;
				ui_bkg_color[1] = g_bkg_color[1] = 0.6f;
				ui_bkg_color[2] = g_bkg_color[2] = 0.5f;
				should_update_bkg = true;
			}ImGui::SameLine();
			if (ImGui::Button("White")) {
				ui_bkg_color[0] = g_bkg_color[0] = 1.f;
				ui_bkg_color[1] = g_bkg_color[1] = 1.f;
				ui_bkg_color[2] = g_bkg_color[2] = 1.f;
				should_update_bkg = true;
			}

			
			ImGui::SliderFloat("Fovy", &g_fovy, 10, 60);
			if (g_fovy != ui_fovy) {
				ui_fovy = g_fovy;
				glMatrixMode(GL_PROJECTION);//
				glLoadIdentity();
				gluPerspective(g_fovy, (float)winWidth / (float)winHeight, 0.01f, 20.f);
				glMatrixMode(GL_MODELVIEW);
			}
			ImGui::SliderFloat("Anisotropy Scale", &max_show_anistropy, 1.1f, 200.f);
			ImGui::SliderFloat("Depth Scale", &scale_show_depth, 0.1f, 10.f);


#ifdef ENABLE_MSAA
			ImGui::Text("Anti-aliasing");
			ImGui::Checkbox("Use MSAA", &use_msaa);
#endif

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		}

		if (ImGui::CollapsingHeader("Shape")) {
			ImGui::PushID(1);
			if (ImGui::Button("Reset")) {
				faceParams.resetShapeParams();
				should_update_mesh = true;
				should_update_gs = true;
			}
			ImGui::PopID();

			char cbuff[128];
			ImGuiStyle& style = ImGui::GetStyle();
			float child_height = ImGui::GetTextLineHeight() + style.ScrollbarSize + style.WindowPadding.y * 2.0f;
			//float child_height = 10.f;
			child_height *= 3.f;
			ImGuiWindowFlags child_flags = ImGuiWindowFlags_HorizontalScrollbar;
			ImGuiID child_id = ImGui::GetID((void*)(intptr_t)(child_seq++));
			bool child_is_visible = ImGui::BeginChild(child_id, ImVec2(-30, child_height), true, child_flags);

			for (int i = 0; i < faceParams.n_params_shape; i++) {
				sprintf(cbuff, "%d", i);
				ImGui::SliderFloat(cbuff, &faceParams.params_shape[i], -2.f, 2.f);
			}
			ImGui::EndChild();
			should_check_diff = true;
		}

		if (ImGui::CollapsingHeader("Expression")) {
			ImGui::PushID(2);
			if (ImGui::Button("Reset")) {
				faceParams.resetExprParams();
				should_update_mesh = true;
				should_update_gs = true;
			}
			ImGui::PopID();
			char cbuff[128];
			ImGuiStyle& style = ImGui::GetStyle();
			float child_height = ImGui::GetTextLineHeight() + style.ScrollbarSize + style.WindowPadding.y * 2.0f;
			//float child_height = 10.f;
			child_height *= 3.f;
			ImGuiWindowFlags child_flags = ImGuiWindowFlags_HorizontalScrollbar;
			ImGuiID child_id = ImGui::GetID((void*)(intptr_t)(child_seq++));
			bool child_is_visible = ImGui::BeginChild(child_id, ImVec2(-30, child_height), true, child_flags);

			for (int i = 0; i < faceParams.n_params_expr; i++) {
				sprintf(cbuff, "%d", i);
				ImGui::SliderFloat(cbuff, &faceParams.params_expr[i], -5.f, 5.f);
				//ImGui::SliderFloat(cbuff, &faceParams.params_expr[i], -1.5f, 1.5f); 
			}
			ImGui::EndChild();
			should_check_diff = true;
		}

		if (ImGui::CollapsingHeader("Pose")) {
			ImGui::PushID(3);
			if (ImGui::Button("Reset")) {
				faceParams.resetPoseParams();
				should_update_mesh = true;
				should_update_gs = true;
			}
			ImGui::Text("0,1,2 overall rotation");
			ImGui::Text("3,4,5 neck movement");
			ImGui::Text("6,7,8 jaw");
			ImGui::Text("9,10,11 left eye");
			ImGui::Text("12,13,14 right eye");
			ImGui::PopID();
			char cbuff[128];
			ImGuiStyle& style = ImGui::GetStyle();
			float child_height = ImGui::GetTextLineHeight() + style.ScrollbarSize + style.WindowPadding.y * 2.0f;
			//float child_height = 10.f;
			child_height *= 3.f;
			ImGuiWindowFlags child_flags = ImGuiWindowFlags_HorizontalScrollbar;
			ImGuiID child_id = ImGui::GetID((void*)(intptr_t)(child_seq++));
			bool child_is_visible = ImGui::BeginChild(child_id, ImVec2(-30, child_height), true, child_flags);

			for (int i = 0; i < faceParams.n_params_pose; i++) {
				sprintf(cbuff, "%d", i);
				ImGui::SliderFloat(cbuff, &faceParams.params_pose[i], -2.f, 2.f);
			}
			ImGui::EndChild();
			should_check_diff = true;
		}

		if (ImGui::CollapsingHeader("Eyelid")) {
			ImGui::PushID(4);
			if (ImGui::Button("Reset")) {
				faceParams.resetEyelidParams();
				should_update_mesh = true;
				should_update_gs = true;
			}			
			for (int i = 0; i < faceParams.n_params_eyelid; i++) {
				sprintf(cbuff, "%d", i);
				ImGui::SliderFloat(cbuff, &faceParams.params_eyelid[i], -2.f, 2.f);
			}
			ImGui::PopID();
			should_check_diff = true;
		}

		ImGui::End();		
		
#ifdef FPS_GRAPH

		ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_MenuBar);
		{
			float old_size = ImGui::GetFont()->Scale;
			ImGui::GetFont()->Scale *= 2;
			ImGui::PushFont(ImGui::GetFont());

			ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
			//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

			ImGui::GetFont()->Scale = old_size;
			ImGui::PopFont();
			//ImGui::TreePop();
		}
		ImGui::End();

#endif

		//bool x = true;
		//ImGui::ShowDemoWindow(&x);

		ImGui::Render();

		if (should_check_diff) {
			bool flag = faceParams.FindDiff();
			should_update_mesh = should_update_mesh || flag;
			should_update_gs = should_update_gs || flag;
		}

	}

	
}


int input_parser(int argc, char ** argv){
	//// Input command parser
	auto cmdl = argh::parser(argc, argv);
	//// print help
	if(cmdl[{"-h","--help"}]){
		std::cout << "FaceViewer_FLAME" << std::endl << std::endl;
		std::cout << "usage: FaceViewer_FLAME --dataset=DATASET --model=MODEL [arguments]" << std::endl << std::endl;
		std::cout << "Arguments:" << std::endl;
		std::cout << "     --dataset         Path to dataset" << std::endl;
		std::cout << "     --model           Path to model" << std::endl;
		std::cout << "     --load_iter       Iteration of model loaded" << std::endl;
		std::cout << "     --n_test          Number of frames reserved for test set " << std::endl;
		std::cout << "     --winWidth        Displayed window width" << std::endl;
		std::cout << "     --winHeight       Displayed window height" << std::endl;
		return 1;
	}
    //// compulsory 
	if(!(cmdl({"--dataset"}) >> g_dataset_path)){
		std::cerr << "Must provide a valid dataset path! Got '" << cmdl("dataset").str() << "'" << std::endl;
		return 1;
	}
	else
		std::cout << "dataset=" << g_dataset_path << std::endl;

	if(!(cmdl({"--model"}) >> g_model_path)){
		std::cerr << "Must provide a valid model path! Got '" << cmdl("model").str() << "'" << std::endl;
		return 1;
	}
	else
		std::cout << "model=" << g_model_path << std::endl;
	//// optional
	std::string l_load_iter;
	cmdl("load_iter", g_load_iter) >> l_load_iter;
	if(l_load_iter != g_load_iter){
		std::cout << "load_iter=" << l_load_iter << std::endl;
		g_load_iter = l_load_iter;
	}

	int ln_test;
	cmdl("n_test", n_test) >> ln_test;
	if(ln_test != n_test){
		std::cout << "n_test=" << ln_test << std::endl;
		n_test = ln_test;
	}


        int lwinWidth, lwinHeight;
	cmdl("winWidth", winWidth) >> lwinWidth;
	if(lwinWidth != winWidth){
		std::cout << "winWidth=" << lwinWidth << std::endl;
		winWidth = lwinWidth;
	}
	cmdl("winHeight", winHeight) >> lwinHeight;
	if(lwinHeight != winHeight){
		std::cout << "winHeight=" << lwinHeight << std::endl;
		winHeight = lwinHeight;
	}
	return 0;

}

int main(int argc, char** argv) {

	//// Input command parser
	if(input_parser(argc, argv)){
		return 1;
	}
	

	GLFWwindow* window;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2); //OpenGL 2.0
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	glfwWindowHint(GLFW_STENCIL_BITS, 8);

#ifdef ENABLE_MSAA
	glfwWindowHint(GLFW_SAMPLES, MSAA_SAMPLE_NUM);
#endif

	window = glfwCreateWindow(winWidth, winHeight, "Exp vis FLAME", NULL, NULL);

	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback); // mouse or touchpad

	glfwMakeContextCurrent(window);
	//gladLoadGL(glfwGetProcAddress);
#ifdef ENABLE_VSYNC
	glfwSwapInterval(1);
#else
	glfwSwapInterval(0);
#endif
	
	GLuint err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(0);
	}
	///////////// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 130";
	ImGui_ImplOpenGL3_Init(glsl_version);

	///////////////////// Rendering Setup
	init();

	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		framebuffer_size_callback(window, width, height);
	}
	while (!glfwWindowShouldClose(window)) // main loop
	{
		glfwPollEvents();

		drawGui();

		ignoring = false;

		// To reproduce performance of 370fps reported in our paper,
		// please uncomment the line to force updating avatar each frame
		// NOTE: 370fps includes both animation and rendering
		//should_update_mesh = true; should_update_gs = true;

		/////
		display();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}

	destroy();

	return 0;

}
