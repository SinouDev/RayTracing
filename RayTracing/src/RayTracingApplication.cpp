#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Timer.h"

#include "Core/Renderer.h"
#include "Core/Camera1.h"

#include "Material/Metal.h"

#include <memory>
#include <chrono>

//#define BENCHMARK_STATIC_DEFINE
//#define BENCHMARK 
//#include "benchmark/benchmark.h"

//static void StringCreation(benchmark::State& state) {
//	for (auto _ : state)
//		std::string empty_string;
//}

// Register the function as a benchmark


// Define another benchmark
//static void StringCopy(benchmark::State& state) {
//	std::string x = "hello";
//	for (auto _ : state)
//		std::string copy(x);
//}

class RayTracingLayer : public Walnut::Layer
{
public:
	
	RayTracingLayer()
	{
		m_Camera = std::make_shared<Camera>(m_CameraInit[0], m_CameraInit[1], m_CameraInit[2], m_CameraInit[3], m_CameraInit[4], m_CameraInit[5]);
		//BENCHMARK(StringCopy);
		//BENCHMARK(StringCreation);
		m_PreviewRenderer.SetScalingEnabled(true);
		//m_Camera = Camera(glm::vec3{ -2.0f, 2.0f, 1.0f }, glm::vec3{ 0.0f, 0.0f, -1.0f }, 20.0f, 16.0f / 9.0f);
		//m_Camera.LookAt(glm::vec3{ 0.0f, 0.0f, -1.0f });
		//m_Camera.LookFrom(glm::vec3{ -2.0f, 2.0f, 1.0f });
		glm::vec3 lookFrom = glm::vec3{13.0f, 2.0f, 3.0f};
		glm::vec3 lookAt = glm::vec3{ 0.0f, 0.0f, -1.0f };

		m_Camera->LookFrom(lookFrom);
		m_Camera->LookAt(lookAt);

		auto dist_to_focus = glm::length(lookFrom - lookAt);
		auto aperture = 2.0;

		//m_CameraInit[4] = aperture;
		//m_CameraInit[5] = dist_to_focus;
	}

	virtual void OnUpdate(float ts) override
	{
		m_Camera->OnUpdate(ts);
		m_Camera->SetFOV(m_CameraInit[0]);
		m_Camera->SetAperture(m_CameraInit[4]);
		m_Camera->SetFocusDistance(m_CameraInit[5]);
		//m_Camera.SetFOV(m_CameraInit[0]);
		//m_Camera.SetNearClip(m_CameraInit[1]);
		//m_Camera.SetFarClip(m_CameraInit[2]);
	}

	virtual void OnUIRender() override
	{
		ImGui::Begin("Specs");
		//ImGui::Button("Button");
		ImGui::Text("Rendering time: %.3fms", m_LastRenderTime);
		ImGui::Text("Dimention: %dx%d", m_ViewportWidth, m_ViewportHeight);
		//ImGui::Text("Camera position: x: %.3f, y: %.3f, y: %.3f", m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().y);
		ImGui::End();

		ImGui::Begin("Control");
		//ImGui::Button("Button");
		ImGui::Checkbox("Real-time Rendering", &m_RealTimeRendering);
		if (!m_RealTimeRendering)
		{
			//RenderPreview();
			if (!m_Renderer.IsRendering()) {
				if (ImGui::Button("Render"))
					m_Renderer.RenderOnce(m_Camera);
				if (ImGui::Button("Start Rendering"))
					m_Renderer.StartAsyncRender(m_Camera);
				if (ImGui::Button("Clear scene"))
						m_Renderer.ClearScene();
			}
			else {
				if (ImGui::Button("Stop Rendering!"))
					m_Renderer.StopRendering();
				if (m_Renderer.IsClearingOnEachFrame())
				{
					if (ImGui::Button("Disable clear delay"))
					{
						m_Renderer.SetClearOnEachFrame(false);
					}
					ImGui::SliderInt("Clear Delay", &(int32_t&)m_Renderer.GetClearDelay(), 1, 1000);
				}
				else if (ImGui::Button("Enable clear delay"))
				{
					m_Renderer.SetClearOnEachFrame(true);
				}
			}
		}
		ImGui::ColorEdit3("Ray background color", &m_Renderer.GetRayBackgroundColor()[0]);
		ImGui::ColorEdit3("Ray background color1", &m_Renderer.GetRayBackgroundColor1()[0]);
		//ImGui::SliderFloat3("Camera Position", &m_Camera->GetPosition()[0], -10.0, 10.0f, "%.3f");
		//ImGui::SliderFloat3("Camera Direction", &m_Camera->GetDirection()[0], -10.0, 10.0f, "%.3f");
		//ImGui::SliderFloat3("Light Direction", &m_Renderer.GetLightDir()[0], -100.0, 100.0f, "%.3f");
		ImGui::SliderFloat3("Camera FOV-near/farClip", &m_CameraInit[0], 0.1f, 90.0f, "%.3f");
		ImGui::SliderFloat("Camera Aperture", &m_CameraInit[4], 0.0f, 1.0f, "%.6f");
		ImGui::SliderFloat("Camera Focus Distance", &m_CameraInit[5], 0.0f, 20.0f, "%.6f");
		ImGui::SliderInt("Rendering threads", &(int32_t&)m_Renderer.GetThreadCount(), 1, 25);
		ImGui::SliderInt("Sampling rate", &(int32_t&)m_Renderer.GetSamplingRate(), 1, 400);
		ImGui::SliderInt("Ray color depth", &(int32_t&)m_Renderer.GetRayColorDepth(), 0, 50);
		ImGui::SliderFloat("Right sphere reflection", m_Renderer.get_right_sphere()->GetFuzz(), 0.0f, 1.0f, "%.3f");
		ImGui::SliderFloat("Glass sphere refraction", m_Renderer.get_glass_sphere()->GetIndexOfRefraction(), 0.0f, 5.0f, "%.3f");
		ImGui::SliderFloat("Glass sphere radius", m_Renderer.GetGlassSphere()->GetRadius(), -1.0f, 5.0f, "%.3f");
		ImGui::SliderFloat3("Glass sphere position", &m_Renderer.GetGlassSphere()->GetCenter()[0], -5.0f, 5.0f, "%.3f");

		ImGui::End();

		m_PreviewRenderer.GetThreadCount() = m_Renderer.GetThreadCount();
		//*m_PreviewRenderer.GetSamplingRate() = *m_Renderer.GetSamplingRate();
		m_PreviewRenderer.GetRayColorDepth() = m_Renderer.GetRayColorDepth();
		*m_PreviewRenderer.get_right_sphere()->GetFuzz() = *m_Renderer.get_right_sphere()->GetFuzz();
		*m_PreviewRenderer.get_glass_sphere()->GetIndexOfRefraction() = *m_Renderer.get_glass_sphere()->GetIndexOfRefraction();		
		*m_PreviewRenderer.GetGlassSphere()->GetRadius() = *m_Renderer.GetGlassSphere()->GetRadius();
		m_PreviewRenderer.GetGlassSphere()->GetCenter() = m_Renderer.GetGlassSphere()->GetCenter();
		m_PreviewRenderer.GetRayBackgroundColor() = m_Renderer.GetRayBackgroundColor();
		m_PreviewRenderer.GetRayBackgroundColor1() = m_Renderer.GetRayBackgroundColor1();
		m_PreviewRenderer.GetLightDir() = m_Renderer.GetLightDir();

		{

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("Render view");

			m_ViewportWidth = ImGui::GetContentRegionAvail().x;
			m_ViewportHeight = m_ViewportWidth / m_Camera->GetAspectRatio();

			auto image = m_FinalImage;
			if (image)
				ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() }, ImVec2(0, 1), ImVec2(1, 0));

			ImGui::End();
			ImGui::PopStyleVar();

		}

		if(false)
		{

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, m_PaddingCenter);
			ImGui::Begin("Preview");

			//m_PreviewViewportWidth = ImGui::GetContentRegionAvail().x;
			m_PreviewRenderViewportWidth = ImGui::GetWindowWidth();
			m_PreviewRenderViewportHeight = ImGui::GetWindowHeight();
			float s_height = m_PreviewRenderViewportWidth / m_Camera->GetAspectRatio();
			float s_width = m_PreviewRenderViewportHeight * m_Camera->GetAspectRatio();

			m_PaddingCenter = { 0.0f, 0.0f };
			
			if (s_width < m_PreviewRenderViewportWidth)
			{
				m_PreviewRenderViewportWidth = s_width;
				m_PaddingCenter.x = (ImGui::GetWindowWidth() - s_width) / 2.0f;
			}
			else
			{
				m_PreviewRenderViewportHeight = s_height;
				m_PaddingCenter.y = (ImGui::GetWindowHeight() - s_height) / 2.0f;
			}

			

			auto image = m_FinalImage;
			if (image)
				ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() }, ImVec2(0, 1), ImVec2(1, 0));

			ImGui::End();
			ImGui::PopStyleVar();

		}

		Render();

		if(m_FinalImage)
			m_FinalImage->SetData(m_Renderer.GetImageDataBuffer()->Get<uint32_t*>());

		//ImGui::ShowDemoWindow();
		//if (m_RealTimeRendering)
			//m_Renderer.Render(m_Camera);
		//	Render();
		//else
		//	RenderPreview();
	}

	void OnAttach() override
	{
		//Profiler::Get().Begin("main");
	}

	void OnDetach() override
	{
		//Profiler::Get().End();
	}

	void SavePPM(const char* path = "image.ppm")
	{
		m_Renderer.SaveAsPPM(path);
	}

	void SavePNG(const char* path = "image.png")
	{
		m_Renderer.SaveAsPNG(path);
	}

private:

	void RenderPreview()
	{
		//m_PreviewRenderer.OnResize(m_PreviewRenderViewportWidth, m_PreviewRenderViewportHeight);
		m_Camera->OnResize(m_PreviewRenderViewportWidth, m_PreviewRenderViewportHeight);

		//m_PreviewRenderer.Render(m_Camera);

	}

	void Render()
	{
		
		Walnut::Timer timer;

		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_Camera->OnResize(m_ViewportWidth, m_ViewportHeight);

		if (m_FinalImage)
		{
			
			if (m_FinalImage->GetWidth() == m_ViewportWidth && m_FinalImage->GetHeight() == m_ViewportHeight)
				return;
			m_FinalImage->Resize(m_ViewportWidth, m_ViewportHeight);
		}
		else
		{
			m_FinalImage = std::make_shared<Walnut::Image>(m_ViewportWidth, m_ViewportHeight, Walnut::ImageFormat::RGBA);
		}

		m_LastRenderTime = timer.ElapsedMillis();
	}

private:

	std::shared_ptr<Walnut::Image> m_FinalImage;
	ImVec2 m_PaddingCenter{ 0.0f, 0.0f };
	float m_CameraInit[6] = { 20.0f, 0.1f, 100.0f, 16.0f / 9.0f, 0.1f, 10.0f };
	bool m_RealTimeRendering = false;
	std::shared_ptr<Camera> m_Camera;
	Renderer m_Renderer;
	Renderer m_PreviewRenderer;
	float m_LastRenderTime = 0;
	uint32_t m_PreviewRenderViewportWidth = 240;
	uint32_t m_PreviewRenderViewportHeight = 240;
	uint32_t m_PreviewViewportWidth;
	uint32_t m_PreviewViewportHeight;
	uint32_t m_ViewportWidth = 1280;
	uint32_t m_ViewportHeight = 720;
};

void generate_name(const std::string& path, const std::string& extention, std::string& name)
{
	auto t = std::chrono::high_resolution_clock::now();

	auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(t);
	auto minutes = std::chrono::time_point_cast<std::chrono::minutes>(t);
	auto hours   = std::chrono::time_point_cast<std::chrono::hours>(t);

	auto s = seconds.time_since_epoch().count() % 60;
	auto m = (minutes.time_since_epoch().count() + s) % 60;
	auto h = (hours.time_since_epoch().count() + m);

	name = path + "snapshot " + std::to_string(h) + "-" + std::to_string(m)+ "-" + std::to_string(s) + " " + std::to_string(t.time_since_epoch().count()) + "." + extention;
}

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	///benchmark::Initialize(&argc, argv);
	//benchmark::RunSpecifiedBenchmarks();
	//benchmark::Shutdown();

	Walnut::ApplicationSpecification spec;
	spec.Name = "Ray Tracing";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<RayTracingLayer>();

	

	app->SetMenubarCallback([app]()
	{
		std::string path = "Screenshots/";
		RayTracingLayer* exLayer = (RayTracingLayer*)(app->GetLayerStack()[0].get());
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Save ppm", "Ctrl + S"))
			{
				std::string name;
				generate_name(path, std::string("ppm"), name);
				exLayer->SavePPM(name.c_str());
			}
			if (ImGui::MenuItem("Save png", "Ctrl + S"))
			{
				std::string name;
				generate_name(path, std::string("png"), name);
				exLayer->SavePNG(name.c_str());
			}
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}