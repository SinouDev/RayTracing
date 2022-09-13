#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Timer.h"

#include "Core/Renderer.h"
#include "Core/Camera.h"
#include "Core/Ray.h"

#include "Core/Material/Lambertian.h"
#include "Core/Material/Metal.h"
#include "Core/Material/Dielectric.h"
#include "Core/Material/DiffuseLight.h"
#include "Core/Material/Metal.h"

#include "Core/Object/Sphere.h"
#include "Core/Object/MovingSphere.h"
#include "Core/Object/BVHnode.h"
#include "Core/Object/XyRect.h"
#include "Core/Object/XzRect.h"
#include "Core/Object/YzRect.h"
#include "Core/Object/Box.h"
#include "Core/Object/Translate.h"
#include "Core/Object/RotateY.h"
#include "Core/Object/RotateX.h"
#include "Core/Object/RotateZ.h"
#include "Core/Object/ConstantMedium.h"

#include "Core/Texture/CheckerTexture.h"
#include "Core/Texture/Texture2D.h"
#include "Core/Texture/NoiseTexture.h"

#include "Utils/Utils.h"
#include "Utils/Time.h"
#include "Utils/Random.h"

#include <memory>
#include <ctime>
#include <chrono>
#include <cstdio>

#include "GLFW/glfw3.h"

std::shared_ptr<Sphere> sphere6;

void scenes(HittableObjectList&, int32_t = 0);

GLFWwindow* main_window = nullptr;

class RayTracingLayer : public Walnut::Layer
{
public:
	
	RayTracingLayer()
	{
		m_Camera = std::make_shared<Camera>(m_CameraInit[0], m_CameraInit[1], m_CameraInit[2], m_CameraInit[3], m_CameraInit[4], m_CameraInit[5], 0.0f, 0.5f);
		//BENCHMARK(StringCopy);
		//BENCHMARK(StringCreation);
		//m_PreviewRenderer.SetScalingEnabled(true);
		//m_Camera = Camera(glm::vec3{ -2.0f, 2.0f, 1.0f }, glm::vec3{ 0.0f, 0.0f, -1.0f }, 20.0f, 16.0f / 9.0f);
		//m_Camera.LookAt(glm::vec3{ 0.0f, 0.0f, -1.0f });
		//m_Camera.LookFrom(glm::vec3{ -2.0f, 2.0f, 1.0f });
		//glm::vec3 lookFrom = glm::vec3{478.0f, 278.0f, -600.0f};
		//glm::vec3 lookAt = glm::vec3{ -1.5f, 0.0f, 5.0f };

		glm::vec3 lookFrom = glm::vec3{ 278.0f, 278.0f, -800.0f };
		glm::vec3 lookAt = glm::vec3{ 0.0f, 0.0f, 2.0f };

		m_Camera->LookFrom(lookFrom);
		m_Camera->LookAt(lookAt);

		auto dist_to_focus = glm::length(lookFrom - lookAt);
		auto aperture = 2.0;

		//m_CameraInit[4] = aperture;
		//m_CameraInit[5] = dist_to_focus;

		m_ThreadCount = m_Renderer.GetThreadCount();
		HittableObjectList hittableList;
		scenes(hittableList, 5);
		m_Renderer.GetHittableObjectList().Add(std::make_shared<HittableObjectList>(hittableList));
		//m_BVHnode = std::make_shared<BVHnode>(m_Rendererm_HittableObjectList, 0.0f, 2.0f);
	}

	virtual void OnUpdate(float ts) override
	{
		m_Camera->OnUpdate(ts);
		m_Camera->SetFOV(m_CameraInit[0]);
		m_Camera->SetAperture(m_CameraInit[4]);
		m_Camera->SetFocusDistance(m_CameraInit[5]);

		m_Camera->SetFOV(m_CameraInit[0]);
		m_Camera->SetNearClip(m_CameraInit[1]);
		m_Camera->SetFarClip(m_CameraInit[2]);
	}

	virtual void OnUIRender() override
	{
		Walnut::Timer timer;

		bool windowFocused = glfwGetWindowAttrib(main_window, GLFW_FOCUSED); // check if window has focus to prevent unresolved memmory allocation when the window is minimized

		float time = m_Renderer.GetRenderingTime();

		//int32_t ms= time % 1000;
		//int32_t s = time / 1000 % 60;
		//int32_t m = time / 1000 / 60 % 60;
		//int32_t h = time / 1000 / 60 / 60 % 60;

		Utils::Time::TimeComponents timeComponents;

		Utils::Time::GetTime(timeComponents, static_cast<std::time_t>(time));

		ImGui::Begin("Specs");
		//ImGui::Button("Button");
		ImGui::Text("ImGui Rendering time: %.3fms", m_LastRenderTime);
		ImGui::Text("Rendering time: %.03fms(%02d:%02d:%02d.%03d)", time, timeComponents.hours, timeComponents.minutes, timeComponents.seconds, timeComponents.milli_seconds);
		ImGui::Text("Renderer FPS: %.02f", time == 0.0f ? 0.0f : 1000.0f / time);
		ImGui::Text("Camera origin: {%.3f, %.3f, %.3f}", m_Camera->GetPosition().x, m_Camera->GetPosition().y, m_Camera->GetPosition().z);
		ImGui::Text("Camera direction: {%.3f, %.3f, %.3f}", m_Camera->GetDirection().x, m_Camera->GetDirection().y, m_Camera->GetDirection().z);
		ImGui::Text("Dimention: %dx%d", m_ViewportWidth, m_ViewportHeight);
		//ImGui::Text("Camera position: x: %.3f, y: %.3f, y: %.3f", m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().y);
		ImGui::End();

		ImGui::Begin("Control");
		//ImGui::Button("Button");
		ImGui::Checkbox("Real-time Rendering", &m_RealTimeRendering);
		//ImGui::Checkbox("Enable BVHnode", &m_Renderer.GetEnableBVHnode());
		//if (!m_RealTimeRendering)
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
					m_Renderer.StopRendering([]()->void {
					std::cout << "Rendering stopped!\n";
					});
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
		ImGui::Checkbox("Set simple ray mode", &Ray::SimpleRayMode());
		ImGui::SliderFloat("Camera move speed", &m_Camera->GetMoveSpeed(), 1.0f, 180.0f, "%.6f");
		ImGui::ColorEdit3("Ray background color", &m_Renderer.GetRayBackgroundColor()[0]);
		ImGui::ColorEdit3("Ray background color1", &m_Renderer.GetRayBackgroundColor1()[0]);
		//ImGui::SliderFloat3("Camera Position", &m_Camera->GetPosition()[0], -10.0, 10.0f, "%.3f");
		//ImGui::SliderFloat3("Camera Direction", &m_Camera->GetDirection()[0], -10.0, 10.0f, "%.3f");
		//ImGui::SliderFloat("Light Intencity", &dynamic_cast<SolidColorTexture*>(m_Renderer.GetLightDir()->GetEmit().get())->GetColor()[0], 0.0f, 100.0f, "%.3f");
		//ImGui::SliderFloat3("Light Position", &m_Renderer.GetLightSphere()->GetCenter()[0], -100.0, 100.0f, "%.3f");
		//ImGui::SliderFloat("Light Radius", m_Renderer.GetLightSphere()->GetRadius(), 0.001f, 10.0f, "%.3f");
		ImGui::SliderFloat3("Camera FOV-near/farClip", &m_CameraInit[0], 0.1f, 90.0f, "%.3f");
		ImGui::SliderFloat("Camera Aperture", &m_CameraInit[4], 0.0f, 1.0f, "%.6f");
		ImGui::SliderFloat("Camera Focus Distance", &m_CameraInit[5], 0.0f, 20.0f, "%.6f");
		if(!m_Renderer.IsRendering())
			ImGui::SliderInt("Rendering threads", &m_ThreadCount, 1, 25);
		ImGui::SliderInt("Sampling rate", &(int32_t&)m_Renderer.GetSamplingRate(), 1, 10000);
		ImGui::SliderInt("Ray color depth", &(int32_t&)m_Renderer.GetRayColorDepth(), 0, 50);
		//ImGui::SliderFloat("Right sphere reflection", m_Renderer.get_right_sphere()->GetFuzz(), 0.0f, 1.0f, "%.3f");
		//ImGui::SliderFloat("Glass sphere refraction", m_Renderer.get_glass_sphere()->GetIndexOfRefraction(), 0.0f, 5.0f, "%.3f");
		//ImGui::SliderFloat("Glass sphere radius", m_Renderer.GetGlassSphere()->GetRadius(), -1.0f, 5.0f, "%.3f");
		//ImGui::SliderFloat3("Glass sphere position", &m_Renderer.GetGlassSphere()->GetCenter()[0], -5.0f, 5.0f, "%.3f");

		ImGui::End();

		//glm::vec4& color = dynamic_cast<SolidColorTexture*>(m_Renderer.GetLightDir()->GetEmit().get())->GetColor();
		//color = glm::vec4(color.r);

		if (m_ThreadCount != m_Renderer.GetThreadCount())
		{
			m_Renderer.SetWorkingThreads(m_ThreadCount);
		}

		//m_PreviewRenderer.GetThreadCount() = m_Renderer.GetThreadCount();
		//*m_PreviewRenderer.GetSamplingRate() = *m_Renderer.GetSamplingRate();
		//m_PreviewRenderer.GetRayColorDepth() = m_Renderer.GetRayColorDepth();
		//*m_PreviewRenderer.get_right_sphere()->GetFuzz() = *m_Renderer.get_right_sphere()->GetFuzz();
		//*m_PreviewRenderer.get_glass_sphere()->GetIndexOfRefraction() = *m_Renderer.get_glass_sphere()->GetIndexOfRefraction();		
		//*m_PreviewRenderer.GetGlassSphere()->GetRadius() = *m_Renderer.GetGlassSphere()->GetRadius();
		//m_PreviewRenderer.GetGlassSphere()->GetCenter() = m_Renderer.GetGlassSphere()->GetCenter();
		//m_PreviewRenderer.GetRayBackgroundColor() = m_Renderer.GetRayBackgroundColor();
		//m_PreviewRenderer.GetRayBackgroundColor1() = m_Renderer.GetRayBackgroundColor1();
		//m_PreviewRenderer.GetLightDir() = m_Renderer.GetLightDir();

		if(m_RealTimeRendering) 
		{

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("Render view");

			m_ViewportWidth = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
			m_ViewportHeight = static_cast<uint32_t>(m_ViewportWidth / m_Camera->GetAspectRatio());

			const auto& image = m_FinalImage;
			if (image && image->GetData())
				ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() }, ImVec2(0, 1), ImVec2(1, 0));

			ImGui::End();
			ImGui::PopStyleVar();

		}
		else
		{

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, m_PaddingCenter);
			ImGui::Begin("Preview");

			//m_PreviewViewportWidth = ImGui::GetContentRegionAvail().x;
			m_PreviewRenderViewportWidth = static_cast<uint32_t>(ImGui::GetWindowWidth());
			m_PreviewRenderViewportHeight = static_cast<uint32_t>(ImGui::GetWindowHeight());
			float s_height = m_PreviewRenderViewportWidth / m_Camera->GetAspectRatio();
			float s_width = m_PreviewRenderViewportHeight * m_Camera->GetAspectRatio();

			m_PaddingCenter = { 0.0f, 0.0f };
			
			if (s_width < m_PreviewRenderViewportWidth)
			{
				m_PreviewRenderViewportWidth = static_cast<uint32_t>(s_width);
				m_PaddingCenter.x = (ImGui::GetWindowWidth() - s_width) / 2.0f;
			}
			else
			{
				m_PreviewRenderViewportHeight = static_cast<uint32_t>(s_height);
				m_PaddingCenter.y = (ImGui::GetWindowHeight() - s_height) / 2.0f;
			}

			

			const auto& image = m_FinalImage;
			if (image && image->GetData())
				ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() }, ImVec2(0, 1), ImVec2(1, 0));

			ImGui::End();
			ImGui::PopStyleVar();

		}

		Render();

		if (m_FinalImage)
		{
			//float a = Walnut::Application::GetTime() - m_LastDrawTime;
			//if (a >= 0.16f)
			if(m_Renderer.IsRendering() && windowFocused)
			{
				m_FinalImage->SetData(m_Renderer.GetImageDataBuffer()->Get<uint8_t*>());
			}
		}

		//ImGui::ShowDemoWindow();
		//if (m_RealTimeRendering)
			//m_Renderer.Render(m_Camera);
		//	Render();
		//else
		//	RenderPreview();
		m_LastRenderTime = timer.ElapsedMillis();
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

	void Render()
	{

		uint32_t width = m_RealTimeRendering ? m_ViewportWidth : m_PreviewRenderViewportWidth;
		uint32_t height = m_RealTimeRendering ? m_ViewportHeight : m_PreviewRenderViewportHeight;

		m_Renderer.OnResize(width, height, [this]()->void {
			m_Renderer.StartAsyncRender(m_Camera);
		});
		m_Camera->OnResize(width, height);
		

		if (m_FinalImage)
		{
			
			if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
				return;
			m_FinalImage->Resize(width, height);
		}
		else
		{
			m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
		}
		//sphere6->SetCenter(glm::vec3(glm::vec2(m_Camera->GetPosition()), m_Camera->GetPosition().z + 0.16f));
		
	}

private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	ImVec2 m_PaddingCenter{ 0.0f, 0.0f };
	float m_CameraInit[6] = { 40.0f, 0.1f, 100.0f, 1.0f / 1.0f, 0.0f, 10.0f };
	bool m_RealTimeRendering = false;
	std::shared_ptr<Camera> m_Camera;
	Renderer m_Renderer;
	//Renderer m_PreviewRenderer;
	float m_LastRenderTime = 0;
	int32_t m_ThreadCount;
	uint32_t m_PreviewRenderViewportWidth = 240;
	uint32_t m_PreviewRenderViewportHeight = 240;
	uint32_t m_PreviewViewportWidth;
	uint32_t m_PreviewViewportHeight;
	uint32_t m_ViewportWidth = 1280;
	uint32_t m_ViewportHeight = 720;
};

void generate_name(const std::string& path, const std::string& extention, std::string& name)
{
	auto clock_now = std::chrono::system_clock::now();
	auto t = Utils::Time::GetTime(std::chrono::time_point_cast<std::chrono::milliseconds>(clock_now).time_since_epoch().count());
	//auto t = std::chrono::high_resolution_clock::now();
	//
	//auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(t);
	//auto minutes = std::chrono::time_point_cast<std::chrono::minutes>(t);
	//auto hours   = std::chrono::time_point_cast<std::chrono::hours>(t);
	//
	//auto s = seconds.time_since_epoch().count() % 60;
	//auto m = (minutes.time_since_epoch().count() + s) % 60;
	//auto h = (hours.time_since_epoch().count() + m);

	char buffer[200];


	sprintf(buffer, "%ssnapshot %02u-%02u-%02u %llu.%s", path.c_str(), (uint32_t)t.hours, (uint32_t)t.minutes, (uint32_t)t.seconds, t.time, extention.c_str());
	//= path + "snapshot " + std::to_string(t.hours) + "-" + std::to_string(t.minutes) + "-" + std::to_string(t.seconds) + " " + std::to_string(t.time) + "." + extention;


	name = buffer;
}

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	///benchmark::Initialize(&argc, argv);
	//benchmark::RunSpecifiedBenchmarks();
	//benchmark::Shutdown();

	Walnut::ApplicationSpecification spec;
	spec.Name = "Ray Tracing";

	Walnut::Application* app = new Walnut::Application(spec);
	main_window = app->GetWindowHandle();
	app->PushLayer<RayTracingLayer>();

	app->SetMenubarCallback([app]()
	{
		std::string path = "Screenshots/";
		RayTracingLayer* exLayer = dynamic_cast<RayTracingLayer*>(app->GetLayerStack()[0].get());
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

void scenes(HittableObjectList& hittableList, int32_t state)
{

	using MaterialPtr = std::shared_ptr<Material>;
	using SpherePtr = std::shared_ptr<Sphere>;

	std::shared_ptr<Texture> texture2d = std::make_shared<Texture2D>("Resources/sphere.jpg");

	MaterialPtr ground_material = std::make_shared<Lambertian>(texture2d);

	MaterialPtr lightDir = std::make_shared<DiffuseLight>(glm::vec3(90000.0f));
	SpherePtr sun = std::make_shared<Sphere>(glm::vec3(-1000.0f, 0.0f, 0.0f), 1000.0f, lightDir);

	MaterialPtr lightDir1 = std::make_shared<DiffuseLight>(glm::vec3(15.0f));
	SpherePtr lightSphere = std::make_shared<Sphere>(glm::vec3(5.0f, 0.0f, 0.0f), 0.1f, lightDir1);

	hittableList.Add(lightSphere);

	MaterialPtr back_shpere = std::make_shared<Metal>(glm::vec3(0.5f, 0.5f, 0.5f), 0.15f);
	MaterialPtr center_sphere = std::make_shared<Lambertian>(glm::vec3(0.7f, 0.3f, 0.3f));
	MaterialPtr left_sphere = std::make_shared<Metal>(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f);
	MaterialPtr right_sphere = std::make_shared<Metal>(glm::vec3(0.1f, 0.95f, 0.82f), 1.0f);
	MaterialPtr small_sphere = std::make_shared<ShinyMetal>(glm::vec3(1.0f, 0.6f, 0.0f));

	MaterialPtr glass_sphere = std::make_shared<Dielectric>(1.019f);


	SpherePtr sphere1 = std::make_shared<Sphere>(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f, back_shpere);
	SpherePtr sphere2 = std::make_shared<Sphere>(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, center_sphere);
	SpherePtr sphere3 = std::make_shared<Sphere>(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f, left_sphere);
	SpherePtr sphere4 = std::make_shared<Sphere>(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f, right_sphere);
	SpherePtr sphere5 = std::make_shared<Sphere>(glm::vec3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);

	sphere6 = std::make_shared<Sphere>(glm::vec3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);


	SpherePtr glassSphere = std::make_shared<Sphere>(glm::vec3(0.0f, 0.0f, 1.0f), -0.5f, glass_sphere);

	switch (state)
	{
	case 2:
	{
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));
		hittableList.Add(sphere1);
		hittableList.Add(sphere2);
		hittableList.Add(sphere3);
		hittableList.Add(sphere4);
		hittableList.Add(sphere5);
		hittableList.Add(sphere6);
		hittableList.Add(glassSphere);
		break;
	}
	case 0:
	{
		std::shared_ptr<Texture> checkerTexture = std::make_shared<CheckerTexture>(glm::vec3{ 0.0f }, glm::vec3{ 1.0f });
		MaterialPtr ground_material = std::make_shared<Lambertian>(checkerTexture);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));


		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = Utils::Random::RandomFloat();
				glm::vec3 center(a + 0.9f * Utils::Random::RandomDouble(), 0.2f, b + 0.9f * Utils::Random::RandomDouble());

				if ((center - glm::vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
					std::shared_ptr<Material> sphere_material;

					if (choose_mat < 0.7f) {
						// diffuse
						auto albedo = Utils::Random::RandomVec3() * Utils::Random::RandomVec3();
						sphere_material = std::make_shared<Lambertian>(albedo);
						auto center2 = center + glm::vec3(0.0f, Utils::Random::RandomDouble(0.0f, 0.5f), 0.0f);
						hittableList.Add(std::make_shared<MovingSphere>(
							center, center2, 0.0f, 1.0f, 0.2f, sphere_material));

						//hittableList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
					}
					else if (choose_mat < 0.85f) {
						// metal
						auto albedo = Utils::Random::RandomVec3(0.5f, 1.0f);
						float fuzz = Utils::Random::RandomFloat(0.0f, 0.5f);
						sphere_material = std::make_shared<Metal>(albedo, fuzz);
						hittableList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
					}
					else {
						// glass
						sphere_material = std::make_shared<Dielectric>(1.5f);
						hittableList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
					}
				}
			}
		}

		MaterialPtr material1 = std::make_shared<Dielectric>(1.5f);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, material1));

		MaterialPtr material2 = std::make_shared<Lambertian>(glm::vec3(0.4f, 0.2f, 0.1f));
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

		MaterialPtr material3 = std::make_shared<Metal>(glm::vec3(0.7f, 0.6f, 0.5f), 0.0f);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, material3));
		break;
	}

	case 3:
	{
		std::shared_ptr<Texture> noiseTexture = std::make_shared<NoiseTexture>(4.0f);
		MaterialPtr noiseMaterial = std::make_shared<Lambertian>(noiseTexture);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, noiseMaterial));
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, 2.0f, 0.0f), 2.0f, noiseMaterial));

		MaterialPtr lightRect = std::make_shared<DiffuseLight>(glm::vec3(4.0f));
		hittableList.Add(std::make_shared<XyRect>(glm::mat2{ glm::vec2{ 3.0f, 1.0f }, glm::vec2{ 5.0f, 3.0f } }, -2.0f, lightRect));

		break;
	}

	case 4:
	{
		MaterialPtr red = std::make_shared<Lambertian>(glm::vec3(0.65f, 0.05f, 0.05f));
		MaterialPtr white = std::make_shared<Lambertian>(glm::vec3(0.73f, 0.73f, 0.73f));
		MaterialPtr green = std::make_shared<Lambertian>(glm::vec3(0.12f, 0.45f, 0.15f));
		//MaterialPtr light = std::make_shared<DiffuseLight>(glm::vec3(15.0f, 15.0f, 15.0f));

		hittableList.Add(std::make_shared<YzRect>(glm::mat2{ glm::vec2{ 0.0f, 0.0f }, glm::vec2{ 555.0f, 555.0f } }, 555.0f, green));
		hittableList.Add(std::make_shared<YzRect>(glm::mat2{ glm::vec2{ 0.0f, 0.0f }, glm::vec2{ 555.0f, 555.0f } }, 0.0f, red));
		//hittableList.Add(std::make_shared<XzRect>(glm::mat2{ glm::vec2{ 213.0f, 227.0f }, glm::vec2{ 343.0f, 332.0f } }, 554.0f, r.m_LightDir));
		hittableList.Add(std::make_shared<XzRect>(glm::mat2{ glm::vec2{ 0.0f, 0.0f }, glm::vec2{ 555.0f, 555.0f } }, 0.0f, white));
		hittableList.Add(std::make_shared<XzRect>(glm::mat2{ glm::vec2{ 0.0f, 0.0f }, glm::vec2{ 555.0f, 555.0f } }, 555.0f, white));
		hittableList.Add(std::make_shared<XyRect>(glm::mat2{ glm::vec2{ 0.0f, 0.0f }, glm::vec2{ 555.0f, 555.0f } }, 555.0f, white));



		std::shared_ptr<Texture> containterTexture2d = std::make_shared<Texture2D>("Resources/container2.png");
		std::shared_ptr<Texture> containterSpecTexture2d = std::make_shared<Texture2D>("Resources/container2_specular.png");

		//std::shared_ptr<Texture> noiseTexture = std::make_shared<NoiseTexture>(4.0f);

		MaterialPtr lambMaterial = std::make_shared<Lambertian>(containterTexture2d);
		MaterialPtr metalMaterial = std::make_shared<ShinyMetal>(containterSpecTexture2d);
		lambMaterial->AddMaterial(metalMaterial);
		//hittableList.Add(std::make_shared<Box>(glm::vec3(130.0f, 0.0f, 65.0f), glm::vec3(295.0f, 165.0f, 230.0f), lambMaterial));
		//hittableList.Add(std::make_shared<Box>(glm::vec3(265.0f, 0.0f, 295.0f), glm::vec3(430.0f, 330.0f, 460.0f), white));

		std::shared_ptr<HittableObject> box1 = std::make_shared<Box>(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(165.0f, 330.0f, 165.0f), white);
		box1 = std::make_shared<RotateZ>(box1, 10.0f);
		box1 = std::make_shared<Translate>(box1, glm::vec3(265.0f, 0.0f, 295.0f));
		//hittableList.Add(box1);

		std::shared_ptr<HittableObject> box2 = std::make_shared<Box>(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(165.0f, 165.0f, 165.0f), white);
		box2 = std::make_shared<RotateZ>(box2, -18.0f);
		box2 = std::make_shared<Translate>(box2, glm::vec3(130.0f, 0.0f, 65.0f));
		//hittableList.Add(box2);

		hittableList.Add(box1);// std::make_shared<ConstantMedium>(box1, 0.01f, glm::vec3(0.0f, 0.0f, 0.0f)));
		hittableList.Add(box2);// std::make_shared<ConstantMedium>(box2, 0.01f, glm::vec3(1.0f, 1.0f, 1.0f)));

		break;
	}

	case 5:
	{
		HittableObjectList boxes1;
		MaterialPtr ground = std::make_shared<Lambertian>(glm::vec3(0.48f, 0.83f, 0.53f));

		const int boxes_per_side = 20;
		for (int i = 0; i < boxes_per_side; i++) {
			for (int j = 0; j < boxes_per_side; j++) {
				float w = 100.0f;
				float x0 = -1000.0f + i * w;
				float z0 = -1000.0f + j * w;
				float y0 = 0.0f;
				float x1 = x0 + w;
				float y1 = Utils::Random::RandomFloat(1.0f, 101.0f);
				float z1 = z0 + w;

				boxes1.Add(std::make_shared<Box>(glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1), ground));
			}
		}


		hittableList.Add(std::make_shared<BVHnode>(boxes1, 0.0f, 1.0f));

		MaterialPtr light = std::make_shared<DiffuseLight>(glm::vec3(7.0f, 7.0f, 7.0f));
		hittableList.Add(std::make_shared<XzRect>(glm::mat2x2{ glm::vec2(123.0f, 147.0f), glm::vec2(423.0f, 412.0f) }, 554.0f, light));

		glm::vec3 center1(400.0f, 400.0f, 200.0f);
		glm::vec3 center2 = center1 + glm::vec3(30.0f, 0.0f, 0.0f);
		MaterialPtr moving_sphere_material = std::make_shared<Lambertian>(glm::vec3(0.7f, 0.3f, 0.1f));
		hittableList.Add(std::make_shared<MovingSphere>(center1, center2, 0.0f, 1.0f, 50.0f, moving_sphere_material));

		MaterialPtr dielectric = std::make_shared<Dielectric>(1.5f);
		MaterialPtr metal = std::make_shared<Metal>(glm::vec3(0.8f, 0.8f, 0.9f), 0.0f);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(260.0f, 150.0f, 45.0f), 50.0f, dielectric));
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, 150.0f, 145.0f), 50.0f, metal));

		std::shared_ptr<HittableObject> boundary = std::make_shared<Sphere>(glm::vec3(360.0f, 150.0f, 145.0f), 70.0f, dielectric);
		hittableList.Add(boundary);
		hittableList.Add(std::make_shared<ConstantMedium>(boundary, 0.2f, glm::vec3(0.2f, 0.4f, 0.9f)));
		boundary = std::make_shared<Sphere>(glm::vec3(0.0f, 0.0f, 0.0f), 5000.0f, dielectric);
		hittableList.Add(std::make_shared<ConstantMedium>(boundary, 0.0001f, glm::vec3(1.0f, 1.0f, 1.0f)));

		std::shared_ptr<Texture> earthTexture = std::make_shared<Texture2D>("Resources/8081_earthmap10k.jpg");
		MaterialPtr emat = std::make_shared<Lambertian>(earthTexture);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(400.0f, 200.0f, 400.0f), 100.0f, emat));
		std::shared_ptr<Texture> pertext = std::make_shared<NoiseTexture>(0.1f);

		MaterialPtr lambertian = std::make_shared<Lambertian>(pertext);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(220.0f, 280.0f, 300.0f), 80.0f, lambertian));

		HittableObjectList boxes2;
		MaterialPtr white = std::make_shared<Lambertian>(glm::vec3(0.73f, 0.73f, 0.73f));
		int ns = 1000;
		for (int j = 0; j < ns; j++) {
			boxes2.Add(std::make_shared<Sphere>(Utils::Random::RandomVec3(0.0f, 165.0f), 10.0f, white));
		}

		std::shared_ptr<HittableObject> hittableBox = std::make_shared<BVHnode>(boxes2, 0.0f, 1.0f);
		std::shared_ptr<HittableObject> hittable = std::make_shared<RotateY>(hittableBox, 15.0f);
		hittableList.Add(std::make_shared<Translate>(hittable, glm::vec3(-100.0f, 270.0f, 395.0f)));

		break;
	}

	default:
	case 1:
	{
		std::shared_ptr<Texture> checker = std::make_shared<CheckerTexture>(glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f));
		std::shared_ptr<Texture> earthTexture2d = std::make_shared<Texture2D>("Resources/8081_earthmap10k.jpg");
		MaterialPtr sphereMaterial1 = std::make_shared<Lambertian>(earthTexture2d);
		std::shared_ptr<Texture> sphereTexture2d = std::make_shared<Texture2D>("Resources/5672_mars_10k_color.jpg");
		MaterialPtr sphereMaterial2 = std::make_shared<Lambertian>(sphereTexture2d);
		hittableList.Add(std::make_shared<Sphere>(glm::vec3(20.0f, 0.0f, 0.0f), 10.0f, sphereMaterial1));
		//hittableList.Add(std::make_shared<Sphere>(glm::vec3(0.0f,  10.0f, 0.0f), 10.0f, sphereMaterial2));
		break;
	}
	}

}