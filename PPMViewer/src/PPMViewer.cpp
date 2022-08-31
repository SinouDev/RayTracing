#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Timer.h"

#include "Renderer.h"

#include <memory>

class ExampleLayer : public Walnut::Layer
{
public:
	virtual void OnUIRender() override
	{
		ImGui::Begin("Specs");
		//ImGui::Button("Button");
		ImGui::Text("Rendering time: %.3fms", m_LastRenderTime);
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");

		m_ViewportWidth = ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = ImGui::GetContentRegionAvail().y;

		auto image = m_Renderer.GetFinalImage();
		if (image)
			ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() });// , ImVec2(0, 1), ImVec2(1, 0));

		ImGui::End();
		ImGui::PopStyleVar();

		//ImGui::ShowDemoWindow();
		Render();
	}

	void SavePPM(const char* path = "D:/Users/yacin/Desktop/RayTracingLearn/WalnutApp/image.ppm")
	{
		m_Renderer.SaveAsPPM(path);
	}

private:

	void Render()
	{
		Walnut::Timer timer;

		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

		m_Renderer.Render();

		m_LastRenderTime = timer.ElapsedMillis();
	}

private:
	Renderer m_Renderer;
	float m_LastRenderTime = 0;
	uint32_t m_ViewportWidth;
	uint32_t m_ViewportHeight;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "PPM Viewer";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();

	

	app->SetMenubarCallback([app]()
	{
		ExampleLayer* exLayer = (ExampleLayer*)(app->GetLayerStack()[0].get());
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Save as", "Ctrl + S"))
			{
				exLayer->SavePPM();
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