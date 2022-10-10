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
#include "Core/Material/Isotropic.h"

#include "Core/Object/Sphere.h"
#include "Core/Object/MovingSphere.h"
#include "Core/Object/ConstantMedium.h"
#include "Core/Object/BVHnode.h"
#include "Core/Object/Box.h"
#include "Core/Object/XyRect.h"
#include "Core/Object/XzRect.h"
#include "Core/Object/YzRect.h"
#include "Core/Object/Translate.h"
#include "Core/Object/Rotate.h"
#include "Core/Object/RotateX.h"
#include "Core/Object/RotateY.h"
#include "Core/Object/RotateZ.h"


#include "Core/Texture/CheckerTexture.h"
#include "Core/Texture/Texture2D.h"
#include "Core/Texture/NoiseTexture.h"
#include "Core/Texture/SolidColorTexture.h"

#include "Utils/Utils.h"
#include "Utils/Time.h"
#include "Utils/Random.h"
#include "Utils/Math.h"
#include "Utils/Color.h"
#include "Utils/Reference.h"

#include <memory>
#include <ctime>
#include <chrono>
#include <cstdio>

#include "GLFW/glfw3.h"

using Utils::Math::Color3;
using Utils::Math::Color4;

using Utils::Math::Vec3;

using Utils::Math::Point2;
using Utils::Math::Point3;

using Utils::Math::Mat2x2;

void scenes(std::shared_ptr<HittableObjectList>&, Vec3&, Vec3&, int32_t = 0);

class RayTracingLayer : public Walnut::Layer
{
public:
	
	RayTracingLayer()
	{

		InitImGuiStyle();
		
		m_Camera = std::make_shared<Camera>(m_CameraInit[0], m_CameraInit[1], m_CameraInit[2], m_CameraInit[3], m_CameraInit[4], m_CameraInit[5], 0.0f, 0.5f);
		m_hittableList = std::make_shared<HittableObjectList>();
		m_Scene = 5;
		m_PreviousScene = -1;
		m_MaxScenes = 6;
		//BENCHMARK(StringCopy);
		//BENCHMARK(StringCreation);
		//m_PreviewRenderer.SetScalingEnabled(true);
		//m_Camera = Camera(Vec3{ -2.0f, 2.0f, 1.0f }, Vec3{ 0.0f, 0.0f, -1.0f }, 20.0f, 16.0f / 9.0f);
		//m_Camera.LookAt(Vec3{ 0.0f, 0.0f, -1.0f });
		//m_Camera.LookFrom(Vec3{ -2.0f, 2.0f, 1.0f });
		//Vec3 lookFrom = Vec3{478.0f, 278.0f, -600.0f};
		//Vec3 lookAt = Vec3{ -1.5f, 0.0f, 5.0f };

		

		

		//auto dist_to_focus = Utils::Math::Q_Length(lookFrom - lookAt);
		//auto aperture = 2.0;

		//scenes(m_hittableList, m_Camera, m_Scene);

		//m_CameraInit[4] = aperture;
		//m_CameraInit[5] = dist_to_focus;

		m_ThreadCount = m_Renderer.GetThreadCount();
		m_SchedulerMultiplier = m_Renderer.GetSchedulerMultiplier();

		m_Renderer.GetRayAmbientLightColorStart() = Vec3(0.035294117f);
		m_Renderer.GetRayAmbientLightColorEnd() = Vec3(0.035294117f);

		m_Renderer.GetSamplingRate() = 5;
		
		m_Renderer.GetHittableObjectList().Add(m_hittableList);

		m_OldAmbientLightColor = m_Renderer.GetRayAmbientLightColorStart();
		m_OldAmbientLightColorStart = m_Renderer.GetRayAmbientLightColorStart();
		m_OldAmbientLightColorEnd = m_Renderer.GetRayAmbientLightColorEnd();
		//m_BVHnode = std::make_shared<BVHnode>(m_Rendererm_HittableObjectList, 0.0f, 2.0f);

		using namespace Utils;
		
		{
			Reference::Shared<SS> vecPtr = Reference::MakeShared<SS>();
			Reference::Unique<SS> vecPtr1 = Reference::MakeUnique<SS>(100);
			{
				Reference::Shared<SS> vecPtr2 = vecPtr;
		
				{
					Reference::Weak<SS> vecPtrWeak = vecPtr2;
				}
			}
		}
		//vecPtr->x = 0.2f;
	}

	virtual void OnUpdate(float ts) override
	{
		m_Camera->OnUpdate(ts);
		m_Camera->SetFOV(m_CameraInit[0]);
		m_Camera->SetNearClip(m_CameraInit[1]);
		m_Camera->SetFarClip(m_CameraInit[2]);
		m_Camera->SetAspectRatio(m_CameraInit[3]);
		m_Camera->SetAperture(m_CameraInit[4]);
		m_Camera->SetFocusDistance(m_CameraInit[5]);
	}

	virtual void OnUIRender() override
	{
		Walnut::Timer timer;

		//const bool windowFocused = glfwGetWindowAttrib(main_window, GLFW_FOCUSED); // check if window has focus to prevent unresolved memmory allocation when the window is minimized
		//ImDrawData* drawData = ImGui::GetDrawData();
		//const bool windowMinimized = drawData ? (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) : true; // check if window is minimized

		if (m_PreviousScene != m_Scene)
		{
			Vec3 lookFrom = Vec3{ 278.0f, 278.0f, -800.0f };
			Vec3 lookAt = Vec3{ 0.0f, 0.0f, 2.0f };
			m_hittableList->Clear();
			scenes(m_hittableList, lookFrom, lookAt, m_Scene);
			m_Camera->LookFrom(lookFrom);
			m_Camera->LookAt(lookAt);
			m_PreviousScene = m_Scene;
		}

		float time = m_Renderer.GetRenderingTime();

		//int32_t ms= time % 1000;
		//int32_t s = time / 1000 % 60;
		//int32_t m = time / 1000 / 60 % 60;
		//int32_t h = time / 1000 / 60 / 60 % 60;

		Utils::Time::TimeComponents timeComponents;

		Utils::Time::GetTime(timeComponents, static_cast<std::time_t>(time));

		{
			ImGui::Begin("Specs");
			//ImGui::Button("Button");
			ImGui::Text("ImGui Rendering time: %.3fms", m_LastRenderTime);
			ImGui::Text("Rendering time: %.03fms(%02d:%02d:%02d.%03d)", time, timeComponents.hours, timeComponents.minutes, timeComponents.seconds, timeComponents.milli_seconds);
			ImGui::Text("Renderer FPS: %.02f | Working/Max Threads %d/%d", time == 0.0f ? 0.0f : 1000.0f / time, m_Renderer.GetThreadCount(), m_Renderer.GetMaximumThreads());
			ImGui::Text("Scheduler Multiplier: %d", m_SchedulerMultiplier);
			ImGui::Separator();
			ImGui::Text("Camera origin: {%.3f, %.3f, %.3f}", m_Camera->GetPosition().x, m_Camera->GetPosition().y, m_Camera->GetPosition().z);
			ImGui::Text("Camera direction: {%.3f, %.3f, %.3f}", m_Camera->GetDirection().x, m_Camera->GetDirection().y, m_Camera->GetDirection().z);
			ImGui::Text("Dimention: %dx%d", m_ViewportWidth, m_ViewportHeight);
			//ImGui::Text("Camera position: x: %.3f, y: %.3f, y: %.3f", m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().y);
			ImGui::End();
		}

		{
			ImGui::Begin("Control");
			//ImGui::Button("Button");

			//ImGui::Checkbox("Enable BVHnode", &m_Renderer.GetEnableBVHnode());
			//if (!m_RealTimeRendering)
			{
				//RenderPreview();
				ImGui::BeginDisabled(m_Renderer.IsRendering());
				ImGui::Checkbox("Full screen Rendering", &m_RealTimeRendering);
				ImGui::SameLine(); HelpMarker("Open new viewport for fullscreen rendering.");
				ImGui::EndDisabled();
				if (!m_Renderer.IsRendering())
				{
					if (ImGui::Button("Render once"))
						m_Renderer.RenderOnce(m_Camera);
					ImGui::SameLine();
					if (ImGui::Button("Start Real-time rendering"))
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
						ImGui::SameLine(); SliderInt("Clear Delay", &(int32_t&)m_Renderer.GetClearDelay(), m_GlobalIdTracker, 1, 1000);
					}
					else if (ImGui::Button("Enable clear delay"))
					{
						m_Renderer.SetClearOnEachFrame(true);
					}
				}
			}
			ImGui::Separator();
			ImGui::Checkbox("Set simple ray mode", &Ray::SimpleRayMode());
			ImGui::BeginDisabled(m_Renderer.IsRendering());
			{
				ImGui::Separator();
				SliderInt("Rendering threads", &m_ThreadCount, m_GlobalIdTracker, 1, m_Renderer.GetMaximumThreads());
				DragInt("Scheduler multiplier", &m_SchedulerMultiplier, m_GlobalIdTracker, 0.1f, 0, 20, "%.3f");
			}
			{
				SliderInt("Scene", &m_Scene, m_GlobalIdTracker, 0, m_MaxScenes);
			}
			ImGui::EndDisabled();
			ImGui::Separator();
			ImGui::Checkbox("Uniform ambient lighting color", &m_UniformAmbientLightingColor);
			if (m_UniformAmbientLightingColor)
			{
				if (m_UniformAmbientLightingColorOld != m_UniformAmbientLightingColor)
				{
					m_OldAmbientLightColorStart = m_Renderer.GetRayAmbientLightColorStart();
					m_OldAmbientLightColorEnd = m_Renderer.GetRayAmbientLightColorEnd();
					m_Renderer.GetRayAmbientLightColorStart() = m_OldAmbientLightColor;
				}

				ColorEdit3("Ray ambient light color", &m_Renderer.GetRayAmbientLightColorStart()[0], m_GlobalIdTracker);
				m_Renderer.GetRayAmbientLightColorEnd() = m_Renderer.GetRayAmbientLightColorStart();
			}
			else
			{
				if (m_UniformAmbientLightingColorOld != m_UniformAmbientLightingColor)
				{
					m_OldAmbientLightColor = m_Renderer.GetRayAmbientLightColorStart();
					m_Renderer.GetRayAmbientLightColorStart() = m_OldAmbientLightColorStart;
					m_Renderer.GetRayAmbientLightColorEnd() = m_OldAmbientLightColorEnd;
				}

				ColorEdit3("Ray ambient light color start", &m_Renderer.GetRayAmbientLightColorStart()[0], m_GlobalIdTracker);
				ColorEdit3("Ray ambient light color end", &m_Renderer.GetRayAmbientLightColorEnd()[0], m_GlobalIdTracker);
			}
			m_UniformAmbientLightingColorOld = m_UniformAmbientLightingColor;
			ImGui::Separator();
			DragFloat("Camera move speed", &m_Camera->GetMoveSpeed(), m_GlobalIdTracker, 1.0f, 1.0f, 18000.0f, "%.6f");
			SliderFloat3("Camera FOV-near/farClip", &m_CameraInit[0], m_GlobalIdTracker, 0.1f, 90.0f, "%.3f");
			ImGui::BeginDisabled(m_Renderer.IsRendering());
			SliderFloat("Camera Aspect Ratio", &m_CameraInit[3], m_GlobalIdTracker, 0.5f, 2.0f, "%.6f");
			ImGui::EndDisabled();
			SliderFloat("Camera Aperture", &m_CameraInit[4], m_GlobalIdTracker, 0.0f, 1.0f, "%.6f");
			SliderFloat("Camera Focus Distance", &m_CameraInit[5], m_GlobalIdTracker, 0.0f, 20.0f, "%.6f");
			ImGui::Separator();
			DragInt("Sampling rate", &(int32_t&)m_Renderer.GetSamplingRate(), m_GlobalIdTracker, 1.0f, 1, 10000);
			DragInt("Ray color depth", &(int32_t&)m_Renderer.GetRayColorDepth(), m_GlobalIdTracker, 1.0f, 1, 200);

			ImGui::End();
		}

		{
			ImGui::Begin("Objects", nullptr, 0);
			HandleHittbleObjectListView(m_hittableList->GetInstance<HittableObjectList>(), m_ObjectViewIdTracker);
			ImGui::End();
		}

		if (m_ThreadCount != m_Renderer.GetThreadCount())
		{
			m_Renderer.SetWorkingThreads(m_ThreadCount);
		}

		if (m_SchedulerMultiplier != m_Renderer.GetSchedulerMultiplier())
		{
			m_Renderer.SetSchedulerMultiplier(m_SchedulerMultiplier);
		}

		if(m_RealTimeRendering)
		{

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("Render view", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

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
			ImGui::Begin("Preview", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);

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

		m_GlobalIdTracker = 0;
		m_ObjectViewIdTracker = 5000;


		//ImGui::ShowDemoWindow();
		auto data = m_Renderer.GetImageDataBuffer().Data();
		if (m_FinalImage && !Walnut::Application::Get().GetMainWindowMinimized())
			m_FinalImage->SetData(data);

		m_LastRenderTime = timer.ElapsedMillis();

	}
	
	virtual void OnDetach() override
	{
		m_Renderer.StopRendering([this]()->void
			{
				m_Renderer.ClearScene();
				m_hittableList->Clear();
			}
		);
	}

	void SavePPM(const char* path = "image.ppm")
	{
		m_Renderer.SaveAsPPM(path);
	}

	void SavePNG(const char* path = "image.png")
	{
		m_Renderer.SaveAsPNG(path);
	}

	inline bool& GetAlwaysShowDescInObjectList() { return m_AlwaysShowDescInObjectList; }

private:

	void HandleHittbleObjectListView(HittableObjectList* hittableObjectList, int32_t& startId)
	{
		// TODO still working on it
		for (const auto& object : hittableObjectList->GetHittableList())
			HandleHittbleObjectView(object->GetInstance(), startId);
	}

	bool HandleHittbleObjectView(HittableObject* object, int32_t& id, bool popTree = true, bool showTree = true)
	{
		// TODO still working on it
		ImGui::PushID(id++);
		bool treeNodeOpen = false;
		if (object->IsFeatureObject())
			HandleFeatureObjectView(object, id);
		else
		{
			volatile auto type = object->GetType();
			switch (type)
			{
				case HittableObjectTypes::SPHERE:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto sphere = object->GetInstance<Sphere>();
						DragFloat3("Center", &sphere->GetCenter()[0], id);
						DragFloat("Radius", sphere->GetRadius(), id);

						HandleMaterialView(sphere->GetMaterial(), id);

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();

					break;
				}

				case HittableObjectTypes::MOVING_SPHERE:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto movingSphere = object->GetInstance<MovingSphere>();

						DragFloat3("Center0", &movingSphere->GetCenter0()[0], id);
						DragFloat3("Center1", &movingSphere->GetCenter1()[0], id);

						HandleMaterialView(movingSphere->GetMaterial(), id);

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::CONTANT_MEDIUM:
				{
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto constantMedium = object->GetInstance<ConstantMedium>();

						DragFloat("Negative inverse density", &constantMedium->GetNegInverseDensity(), id);

						constantMedium->GetNegInverseDensity() = -Utils::Math::Abs(constantMedium->GetNegInverseDensity());

						HandleMaterialView(constantMedium->GetMaterial(), id);

						bool treeOpen = HandleHittbleObjectView(constantMedium->GetBoundary()->GetInstance(), ++id, false, false);

						if (treeOpen)
						{
							ImGui::TreePop();
						}
						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					break;
				}

				case HittableObjectTypes::BOX:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto box = object->GetInstance<Box>();
						//ImGui::SliderFloat3("", &box->GetCenter()[0], -100.0f, 100.0f, "%.3f");

						//ImGui::Indent();
						HandleHittbleObjectListView(box->GetSides(), ++id);
						//ImGui::Unindent();

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::XY_RECT:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto xyRect = object->GetInstance<XyRect>();

						DragFloat2("Point 0", &xyRect->GetPositions()[0][0], id);
						DragFloat2("Point 1", &xyRect->GetPositions()[1][0], id);
						DragFloat("k", &xyRect->GetK(), id);

						DragFloat3("Translate", &xyRect->GetObjectTranslate()[0], id);
						DragFloat3("Rotate", &xyRect->GetObjectRotate()[0], id);


						HandleMaterialView(xyRect->GetMaterial(), id);

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::XZ_RECT:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto xzRect = object->GetInstance<XzRect>();

						DragFloat2("Point 0", &xzRect->GetPositions()[0][0], id, 1.0f, 0.0f, 0.0f, nullptr, "z: %.3f");
						DragFloat2("Point 1", &xzRect->GetPositions()[1][0], id, 1.0f, 0.0f, 0.0f, nullptr, "z: %.3f");

						HandleMaterialView(xzRect->GetMaterial(), id);

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::YZ_RECT:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto yzRect = object->GetInstance<YzRect>();

						DragFloat2("Point 0", &yzRect->GetPositions()[0][0], id, 1.0f, 0.0f, 0.0f, "y: %.3f", "z: %.3f");
						DragFloat2("Point 1", &yzRect->GetPositions()[1][0], id, 1.0f, 0.0f, 0.0f, "y: %.3f", "z: %.3f");

						HandleMaterialView(yzRect->GetMaterial(), id);

						if (popTree)
						{
							ImGui::Separator();
							ImGui::TreePop();
						}

						treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::OBJECT_LIST:
				{
					HandleHittbleObjectListView(object->GetInstance<HittableObjectList>(), ++id);
					break;
				}

				default:
				case HittableObjectTypes::UNKNOWN_OBJECT:
					break;
				}
		}
		ImGui::PopID();
		return treeNodeOpen;
	}

	void HandleFeatureObjectView(HittableObject* object, int32_t& id)
	{
		if (object->IsFeatureObject())
			return;

		switch (object->GetType())
		{
			case HittableObjectTypes::TRANSLATE:
				{
					//ImGui::PushID(++id);
					auto translate = object->GetInstance<Translate>();

					if (translate->GetObject()->GetInstance()->IsFeatureObject())
					{
						HandleFeatureObjectView(translate->GetObject()->GetInstance(), ++id);
					}
					
					//bool treeOpen = 
					
					else
					{
						bool treeOpen = HandleHittbleObjectView(translate->GetObject()->GetInstance(), ++id, false);
						ImGui::TextUnformatted(object->GetName());
						DragFloat3("Position", &translate->GetTranslatePosition()[0], id);
						if (treeOpen)
						{
							ImGui::TreePop();
						}
					}
					//ImGui::PopID();
					break;
				}
#if 0
				case HittableObjectTypes::ROTATE:
				{
					//ImGui::PushID(++id);
					auto rotate = object->GetInstance<Rotate>();
					auto obj = rotate->GetObject()->GetInstance();
					//obj->GetHittable() = rotate->IsHittable();
					//if (EnableTreeNode(obj->GetName(), &rotate->GetHittable(), id, rotate->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(obj->GetType())))

					if (HandleHittbleObjectView(obj, ++id, false))
					{
						ImGui::Text(rotate->GetName());
						ImGui::Separator();
						//SliderAngle3("Angle", &rotate->GetAngle()[0], id);
						//DragFloat3("Angle", &rotate->GetAngle()[0], "x: %.3f", "y: %.3f", "z: %.3f", id);
						//rotate->RotateAxis();

						HandleHittbleObjectView(rotate->GetRotateX()->GetInstance(), ++id, false, false);
						HandleHittbleObjectView(rotate->GetRotateY()->GetInstance(), ++id, false, false);
						HandleHittbleObjectView(rotate->GetRotateZ()->GetInstance(), ++id, false, false);

						//ImGui::Checkbox("Node Content:", &obj->GetHittable());
						//ImGui::Separator();
						//
						//ImGui::BeginDisabled(!obj->IsHittable());
						//ImGui::Indent();
						//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);
						//ImGui::Unindent();
						//ImGui::EndDisabled();

						//if (treeOpen)
						//{
						//	ImGui::TreePop();
						//}
						//if (popTree)
						//{
						//	ImGui::TreePop();
						//}
						//treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::ROTATE_X:
				{
					//ImGui::PushID(++id);
					auto rotateX = object->GetInstance<RotateX>();
					auto obj = rotateX->GetObject()->GetInstance();
					//if (EnableTreeNode(obj->GetName(), &obj->GetHittable(), id, obj->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(obj->GetType())))
					if (HandleHittbleObjectView(obj, ++id, false))
					{
						ImGui::Text(rotateX->GetName());
						ImGui::Separator();
						SliderAngle("AngleX", &rotateX->GetAngle(), id);
						rotateX->Rotate();

						//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);

						//if (treeOpen)
						{
							//if (popTree)
							//{
							//	ImGui::TreePop();
							//}
							//treeNodeOpen = showTree;
						}
						//ImGui::TreePop();
					}
					//ImGui::PopID();
					break;
				}

				case HittableObjectTypes::ROTATE_Y:
				{
					//ImGui::PushID(++id);
					auto rotateY = object->GetInstance<RotateY>();
					auto obj = rotateY->GetObject()->GetInstance();
					//if (EnableTreeNode(obj->GetName(), &obj->GetHittable(), id, obj->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(obj->GetType())))
					if (HandleHittbleObjectView(obj, ++id, false))
					{
						ImGui::Text(rotateY->GetName());
						ImGui::Separator();
						ImGui::SliderAngle("AngleY", &rotateY->GetAngle());
						rotateY->Rotate();

						//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);

						//if (treeOpen)
						{
							//if (popTree)
							//{
							//	ImGui::TreePop();
							//}
							//treeNodeOpen = showTree;
						}
						//ImGui::TreePop();
					}
					//ImGui::PopID();
					break;
				}
#endif
				case HittableObjectTypes::ROTATE_Z:
				{
					//ImGui::PushID(++id);
					auto rotateZ = object->GetInstance<RotateZ>();
					auto obj = rotateZ->GetObject()->GetInstance();

					if (rotateZ->GetObject()->GetInstance()->IsFeatureObject())
					{
						HandleFeatureObjectView(rotateZ->GetObject()->GetInstance(), ++id);
					}

					//bool treeOpen = 

					else
					{
						bool treeOpen = HandleHittbleObjectView(rotateZ->GetObject()->GetInstance(), ++id, false);
						ImGui::Text(rotateZ->GetName());
						ImGui::Separator();
						ImGui::SliderAngle("AngleZ", &rotateZ->GetAngle());
						rotateZ->Rotate();
						if (treeOpen)
						{
							ImGui::TreePop();
						}
					}

					//if (EnableTreeNode(obj->GetName(), &obj->GetHittable(), id, obj->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(obj->GetType())))
					//if (HandleHittbleObjectView(obj, ++id, false))
					//{
					//	ImGui::Text(rotateZ->GetName());
					//	ImGui::Separator();
					//	ImGui::SliderAngle("AngleZ", &rotateZ->GetAngle());
					//	rotateZ->Rotate();
					//
					//	//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);
					//
					//	//if (treeOpen)
					//	{
					//		//if (popTree)
					//		//{
					//		//	ImGui::TreePop();
					//		//}
					//		//treeNodeOpen = showTree;
					//	}
					//	//ImGui::TreePop();
					//}
					////ImGui::PopID();
					break;
				}
#if 0
				case HittableObjectTypes::BVH_NODE:
				{
					//ImGui::PushID(++id);
					if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), false, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(object->GetType())))
					{
						auto bvhNode = object->GetInstance<BVHnode>();

						HandleBVHnode(bvhNode, id);

						//if (popTree)
						//{
						//	ImGui::TreePop();
						//}
						//treeNodeOpen = showTree;
					}
					//ImGui::PopID();
					break;
				}
#endif
		}
	}

	void HandleMaterialView(Material* material, int32_t& id)
	{
		ImGui::PushID(++id);
		switch (material->GetType())
		{
			case MaterialType::DIELECTRIC:
			{
				auto dielectric = material->GetInstance<Dielectric>();
				ImGui::Text("Dielectric");
				DragFloat("Index of refraction", &dielectric->GetIndexOfRefraction(), id, 0.0001f);
				break;
			}
			case MaterialType::DIFFUSE_LIGHT:
			{
				auto diffuseLight = material->GetInstance<DiffuseLight>();
				ImGui::Text("Diffuse Light");
				DragFloat("Brightness", &diffuseLight->GetBrightness(), id, 0.0001f);
				HandleTextureView(diffuseLight->GetEmit()->GetInstance(), ++id);
				break;
			}
			case MaterialType::ISOTROPIC:
			{
				auto isotropic = material->GetInstance<Isotropic>();
				ImGui::Text("Isotropic");
				HandleTextureView(isotropic->GetAlbedo()->GetInstance(), ++id);
				break;
			}
			case MaterialType::LAMBERTIAN:
			{
				auto lambertian = material->GetInstance<Lambertian>();
				ImGui::Text("Lambertian");
				HandleTextureView(lambertian->GetAlbedo()->GetInstance(), ++id);
				break;
			}
			case MaterialType::SHINY_METAL:
			{

			}
			case MaterialType::METAL:
			{

				break;
			}
			

			case MaterialType::UNKNOWN_MATERIAL:
			default:
				break;
		}
		ImGui::PopID();
	}

	void HandleTextureView(Texture* texture, int32_t& id)
	{
		ImGui::PushID(++id);
		switch (texture->GetType())
		{
			case TextureType::CHECKER_TEXTURE:
			{
				auto checker = texture->GetInstance<CheckerTexture>();
				DragInt("Size", (int32_t*) & checker->GetSize(), id);
				HandleTextureView(checker->GetEven()->GetInstance(), ++id);
				HandleTextureView(checker->GetOdd()->GetInstance(), ++id);
				break;
			}
			case TextureType::NOISE_TEXTURE:
			{
				//auto noise = texture->GetInstance<NoiseTexture>();
				
				break;
			}
			case TextureType::SOLID_COLOR_TEXTURE:
			{
				auto solidColor = texture->GetInstance<SolidColorTexture>();
				ColorEdit3("Color", &solidColor->GetColor()[0], id);
				break;
			}
			case TextureType::TEXTURE_2D:
			{
				auto texture2d = texture->GetInstance<Texture2D>();
				ImGui::Text("File: %s", texture2d->GetFileName());
				break;
			}

			default:
			case TextureType::UNKNOWN_TEXTURE:
				break;
		}
		ImGui::PopID();
	}

	void HandleBVHnode(BVHnode* bvhNode, int32_t& id)
	{
		HandleBVHnodeSidesView(bvhNode->GetLeft()->GetInstance(), id);
		HandleBVHnodeSidesView(bvhNode->GetRight()->GetInstance(), id);
	}

	void HandleBVHnodeSidesView(HittableObject* object, int32_t& id)
	{
		if (object->GetType() == BVH_NODE)
		{
			HandleBVHnode(object->GetInstance<BVHnode>(), id);
			return;
		}
		HandleHittbleObjectView(object, ++id);
	}

	bool EnableTreeNode(const char* label, bool* check, int32_t& id, bool condition = true, bool showTree = true, bool alwaysShowDesc = false, const char* descFrm = nullptr, ...)
	{
		if (!showTree)
			return true;
		ImGui::PushID(++id);
		ImGui::Checkbox("", check);
		ImGui::PopID();
		ImGui::SameLine();
		if (!condition)
		{
			ImGui::BeginDisabled();
			if (ImGui::TreeNodeEx((void*)(intptr_t)id, ImGuiTreeNodeFlags_SpanAvailWidth, label))
				ImGui::TreePop();
			ImGui::EndDisabled();
		}
		bool tree = condition && ImGui::TreeNodeEx((void*)(intptr_t)id, ImGuiTreeNodeFlags_SpanAvailWidth, label);
		if (descFrm)
		{
			va_list args;
			va_start(args, descFrm);
			ImGui::SameLine();
			if (alwaysShowDesc)
			{
				std::string s = "(";
				s.append(descFrm);
				s.append(")");
				ImGui::TextDisabledV(s.c_str(), args);
			}
			else
				HelpMarkerV(descFrm, args);
			va_end(args);
		}
		return tree;
	}

	void ColorEdit3(const char* label, float* color, int32_t& id, ImGuiBackendFlags flags = 0)
	{
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::TextUnformatted(label);
		ImGui::SameLine(); ImGui::ColorEdit3("", color, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderAngle(const char* label, float* deg, int32_t& id, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format == nullptr)
			format = "x: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderAngle("", deg, -360.0f, 360.0f, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderAngle2(const char* label, float* deg, int32_t& id, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderAngle("", deg, -360.0f, 360.0f, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderAngle("", deg + 1, -360.0f, 360.0f, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderAngle3(const char* label, float* deg, int32_t& id, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderAngle("", deg, -360.0f, 360.0f, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderAngle("", deg + 1, -360.0f, 360.0f, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderAngle("", deg + 2, -360.0f, 360.0f, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragFloat(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragFloat("", value, speed, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragFloat2(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragFloat("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragFloat("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragFloat3(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragFloat("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragFloat("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragFloat("", value + 2, speed, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragInt(const char* label, int32_t* value, int32_t& id, float speed = 1.0, int32_t min = 0, int32_t max = 0, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format == nullptr)
			format = "%.3f";
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragInt("", value, speed, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragInt2(const char* label, int32_t* value, int32_t& id, float speed = 1.0f, int32_t min = 0, int32_t max = 0, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragInt("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragInt("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void DragInt3(const char* label, int32_t* value, int32_t& id, float speed = 1.0f, int32_t min = 0, int32_t max = 0, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::DragInt("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragInt("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::DragInt("", value + 2, speed, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderFloat(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderFloat("", value, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderFloat2(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderFloat("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderFloat("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderFloat3(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderFloat("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderFloat("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderFloat("", value + 2, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderInt(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderInt("", value, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderInt2(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderInt("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderInt("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void SliderInt3(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); ImGui::SliderInt("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderInt("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); ImGui::SliderInt("", value + 2, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
	}

	void Render()
	{

		uint32_t width = m_RealTimeRendering ? m_ViewportWidth : m_PreviewRenderViewportWidth;
		uint32_t height = m_RealTimeRendering ? m_ViewportHeight : m_PreviewRenderViewportHeight;
		
		m_Renderer.OnResize(width, height, [this, width, height](bool wasRendering)->void {
			
			if(wasRendering)
				m_Renderer.StartAsyncRender(m_Camera);

			if (m_FinalImage)
			{
				if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
					return;
				m_FinalImage->Resize(width, height);
			}
			else
			{
				m_FinalImage = std::make_unique<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
			}

		});
		m_Camera->OnResize(width, height);
		

		
		//sphere6->SetCenter(Point3(Point2(m_Camera->GetPosition()), m_Camera->GetPosition().z + 0.16f));
		
	}

	void InitImGuiStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();

		// set item style

		style.FrameRounding = 12.0f;
		style.WindowRounding = 12.0f;
		style.ChildRounding = 12.0f;
		style.GrabRounding = 12.0f;
		style.PopupRounding = 12.0f;
		style.ScrollbarRounding = 12.0f;
		style.TabRounding = 12.0f;
		style.WindowTitleAlign.x = 0.5f;
		style.FramePadding.x = 7.0f;

		// setup color style
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_FrameBg].x,            0x8029298A);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_FrameBgHovered].x,     0xFA424266);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_FrameBgActive].x,      0xFA4242AB);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TitleBgActive].x,      0x7A2929FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_CheckMark].x,          0xFA4242FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_SliderGrab].x,         0xE03D3DFF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_SliderGrabActive].x,   0xFA4242FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_Button].x,             0xFA424266);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_ButtonHovered].x,      0xFA4242FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_ButtonActive].x,       0xFA0F0FFF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_Header].x,             0xFA4242FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_HeaderHovered].x,      0xFA4242CC);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_HeaderActive].x,       0xFA4242FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_Separator].x,          0x806E6E80);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_SeparatorHovered].x,   0xBF1A1AC7);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_SeparatorActive].x,    0xBF1A1AFF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_ResizeGrip].x,         0xFA424233);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_ResizeGripHovered].x,  0xFA4242AB);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_ResizeGripActive].x,   0xFA4242F2);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_Tab].x,                0x942E2EDC);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TabHovered].x,         0xFA4242CC);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TabActive].x,          0xAD3333FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TabUnfocused].x,       0x261111F8);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TabUnfocusedActive].x, 0x6C2323FF);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_DockingPreview].x,     0xFA4242B3);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_TextSelectedBg].x,     0xFA424259);
		Utils::Color::RGBAtoColorFloats(&style.Colors[ImGuiCol_NavHighlight].x,       0xFA4242FF);

	}

	// Copied from imgui_demo.cpp
	// Helper to display a little (?) mark which shows a tooltip when hovered.
	// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
	static void HelpMarker(const char* desc, ...)
	{
		va_list args;
		va_start(args, desc);
		HelpMarkerV(desc, args);
		va_end(args);
	}
	static void HelpMarkerV(const char* desc, va_list args)
	{
		ImGui::TextDisabled("(?)");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextV(desc, args);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

private:

	std::shared_ptr<HittableObjectList> m_hittableList;
	std::unique_ptr<Walnut::Image> m_FinalImage;
	std::shared_ptr<Camera> m_Camera;

	float m_CameraInit[6] = { 40.0f, 0.1f, 100.0f, 16.0f / 9.0f, 0.0f, 10.0f };
	float m_LastRenderTime = 0;

	Renderer m_Renderer;

	ImVec2 m_PaddingCenter{ 0.0f, 0.0f };

	Color3 m_OldAmbientLightColorStart{ 0.0f };
	Color3 m_OldAmbientLightColorEnd{ 0.0f };
	Color3 m_OldAmbientLightColor{ 0.0f };

	int32_t m_GlobalIdTracker = 0, m_ObjectViewIdTracker = 5000;
	int32_t m_Scene, m_PreviousScene, m_MaxScenes;
	int32_t m_ThreadCount = 0;
	int32_t m_SchedulerMultiplier = 0;

	uint32_t m_PreviewRenderViewportWidth = 240;
	uint32_t m_PreviewRenderViewportHeight = 240;
	uint32_t m_PreviewViewportWidth;
	uint32_t m_PreviewViewportHeight;
	uint32_t m_ViewportWidth = 1280;
	uint32_t m_ViewportHeight = 720;

	bool m_RealTimeRendering = false;
	bool m_UniformAmbientLightingColor = false;
	bool m_UniformAmbientLightingColorOld = false;

	bool m_AlwaysShowDescInObjectList = false;
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

#define ENABLE_TEST 0

#if ENABLE_TEST
void test();
#endif

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{

	Walnut::ApplicationSpecification spec;
	spec.Name = "Ray Tracing";

	Walnut::Application* app = new Walnut::Application(spec);
	
#if ENABLE_TEST
	test();
	return app;
#endif
	auto exLayer = app->PushAndGetLayer<RayTracingLayer>();

	app->SetMenubarCallback([app, exLayer]()
	{
		std::string path = "Screenshots/";
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
		if (ImGui::BeginMenu("View"))
		{
			ImGui::Checkbox("Always show object description", &exLayer->GetAlwaysShowDescInObjectList());
			ImGui::EndMenu();
		}
	});
	return app;
}

void scenes(std::shared_ptr<HittableObjectList>& hittableList, Vec3& lookFrom, Vec3& lookAt, int32_t scene)
{

	using MaterialPtr       = std::shared_ptr<Material>;
	using SpherePtr         = std::shared_ptr<Sphere>;
	using TexturePtr        = std::shared_ptr<Texture>;
	using HittableObjectPtr = std::shared_ptr<HittableObject>;



	switch (scene)
	{
		case 6:
		{
			MaterialPtr lightDir = std::make_shared<DiffuseLight>(Color3(1.0f), 900000.0f);
			SpherePtr sun = std::make_shared<Sphere>(Point3(8000.0f, 1000000.0f, 0.0f), 5000.0f, lightDir);

			MaterialPtr houseMaterial = std::make_shared<Lambertian>(Color4(1.0f, 0.0f, 1.0f, 1.0f));
			HittableObjectPtr mainHouse = std::make_shared<Box>(Point3(0.0f, 0.0f, 0.0f), Point3(555.0f, 555.0f, 555.0f), houseMaterial);

			MaterialPtr ground_material = std::make_shared<Lambertian>(Color3(0.5f, 0.5f, 0.5f));
			hittableList->Add(std::make_shared<Sphere>(Point3(-255.0f, -200000.0f, 0.0f), 200000.0f, ground_material));

			MaterialPtr material3 = std::make_shared<Dielectric>(5.8f);
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, 0.0f, 0.0f), 1000000.0f, material3));

			hittableList->Add(mainHouse);
			hittableList->Add(sun);
			break;
		}
		case 2:
		{
			TexturePtr texture2d = std::make_shared<Texture2D>("Resources/sphere.jpg");

			MaterialPtr ground_material = std::make_shared<Lambertian>(texture2d);

			MaterialPtr lightDir = std::make_shared<DiffuseLight>(Color3(1.0f), 90000.0f);
			SpherePtr sun = std::make_shared<Sphere>(Point3(-1000.0f, 0.0f, 0.0f), 1000.0f, lightDir);
			sun->SetName("Sun");

			MaterialPtr lightDir1 = std::make_shared<DiffuseLight>(Color3(1.0f), 15.0f);
			SpherePtr lightSphere = std::make_shared<Sphere>(Point3(5.0f, 0.0f, 0.0f), 0.1f, lightDir1);

			lightSphere->SetName("Light sphere");

			hittableList->Add(sun);
			hittableList->Add(lightSphere);

			MaterialPtr back_shpere = std::make_shared<Metal>(Color3(0.5f, 0.5f, 0.5f), 0.15f);
			MaterialPtr center_sphere = std::make_shared<Lambertian>(Color3(0.7f, 0.3f, 0.3f));
			MaterialPtr left_sphere = std::make_shared<Metal>(Color3(0.8f, 0.8f, 0.8f), 0.3f);
			MaterialPtr right_sphere = std::make_shared<Metal>(Color3(0.1f, 0.95f, 0.82f), 1.0f);
			MaterialPtr small_sphere = std::make_shared<ShinyMetal>(Color3(1.0f, 0.6f, 0.0f));
			MaterialPtr glass_sphere = std::make_shared<Dielectric>(1.019f);

			SpherePtr sphere1 = std::make_shared<Sphere>(Point3(0.0f, -100.5f, -1.0f), 100.0f, back_shpere);
			SpherePtr sphere2 = std::make_shared<Sphere>(Point3(0.0f, 0.0f, -1.0f), 0.5f, center_sphere);
			SpherePtr sphere3 = std::make_shared<Sphere>(Point3(-1.0f, 0.0f, -1.0f), 0.5f, left_sphere);
			SpherePtr sphere4 = std::make_shared<Sphere>(Point3(1.0f, 0.0f, -1.0f), 0.5f, right_sphere);
			SpherePtr sphere5 = std::make_shared<Sphere>(Point3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);
			SpherePtr sphere6 = std::make_shared<Sphere>(Point3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);
			SpherePtr glassSphere = std::make_shared<Sphere>(Point3(0.0f, 0.0f, 1.0f), -0.5f, glass_sphere);

			sphere1->SetName("Back Sphere");
			sphere2->SetName("Center Sphere");
			sphere3->SetName("Left Sphere");
			sphere4->SetName("Right Sphere");
			sphere5->SetName("Small Sphere 1");
			sphere6->SetName("Small Sphere 2");
			glassSphere->SetName("Glass Sphere");

			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));
			hittableList->Add(sphere1);
			hittableList->Add(sphere2);
			hittableList->Add(sphere3);
			hittableList->Add(sphere4);
			hittableList->Add(sphere5);
			hittableList->Add(sphere6);
			hittableList->Add(glassSphere);
			break;
		}
		case 0:
		{
			HittableObjectList list1;
			HittableObjectList list2;
			HittableObjectList list3;
			HittableObjectList list4;
			//std::shared_ptr<Texture> checkerTexture = std::make_shared<CheckerTexture>(Color3{ 0.0f }, Color3{ 1.0f });
			MaterialPtr ground_material = std::make_shared<Lambertian>(Color3(0.5f, 0.5f, 0.5f));
			auto ground = std::make_shared<Sphere>(Point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material);
			ground->SetName("Ground");
			list4.Add(ground);


			for (int32_t a = -11; a < 11; a++)
			{
				for (int32_t b = -11; b < 11; b++) 
				{
					float choose_mat = Utils::Random::RandomFloat();
					Point3 center(a + 0.9f * Utils::Random::RandomDouble(), 0.2f, b + 0.9f * Utils::Random::RandomDouble());

					if (Utils::Math::Q_Length(center - Point3(4.0f, 0.2f, 0.0f)) > 0.9f) {
						MaterialPtr sphere_material;

						if (choose_mat < 0.7f) {
							// diffuse
							Color3 albedo = Utils::Random::RandomVec3() * Utils::Random::RandomVec3();
							sphere_material = std::make_shared<Lambertian>(albedo);
							Point3 center2 = center + Point3(0.0f, Utils::Random::RandomDouble(0.0f, 0.5f), 0.0f);
							auto ball = std::make_shared<MovingSphere>(center, center2, 0.0f, 1.0f, 0.2f, sphere_material);
							ball->SetName("Diffuse ball");
							list1.Add(ball);

							//hittableList->Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
						}
						else if (choose_mat < 0.85f) {
							// metal
							Color3 albedo = Utils::Random::RandomVec3(0.5f, 1.0f);
							float fuzz = Utils::Random::RandomFloat(0.0f, 0.5f);
							sphere_material = std::make_shared<Metal>(albedo, fuzz);
							auto ball = std::make_shared<Sphere>(center, 0.2f, sphere_material);
							ball->SetName("Metal ball");
							list2.Add(ball);
						}
						else {
							// glass
							sphere_material = std::make_shared<Dielectric>(1.5f);
							auto ball = std::make_shared<Sphere>(center, 0.2f, sphere_material);
							ball->SetName("Glass ball");
							list3.Add(ball);
						}
					}
				}
			}

			MaterialPtr material1 = std::make_shared<Dielectric>(1.5f);
			auto glassSphere = std::make_shared<Sphere>(Point3(0.0f, 1.0f, 0.0f), 1.0f, material1);
			glassSphere->SetName("Glass sphere");
			list4.Add(glassSphere);

			MaterialPtr material2 = std::make_shared<Lambertian>(Color3(0.4f, 0.2f, 0.1f));
			auto diffuseSphere = std::make_shared<Sphere>(Point3(-4.0f, 1.0f, 0.0f), 1.0f, material2);
			diffuseSphere->SetName("Diffuse sphere");
			list4.Add(diffuseSphere);

			MaterialPtr material3 = std::make_shared<Metal>(Color3(0.7f, 0.6f, 0.5f), 0.0f);
			auto metalSphere = std::make_shared<Sphere>(Point3(4.0f, 1.0f, 0.0f), 1.0f, material3);
			metalSphere->SetName("Matal sphere");
			list4.Add(metalSphere);

			auto diffuseSpheres = std::make_shared<BVHnode>(list1, 0.0f, 1.0f);
			auto metalSpheres = std::make_shared<BVHnode>(list2, 0.0f, 1.0f);
			auto glassSpheres = std::make_shared<BVHnode>(list3, 0.0f, 1.0f);
			auto globalSpheres = std::make_shared<BVHnode>(list4, 0.0f, 1.0f);

			diffuseSpheres->SetName("Diffuse Spheres");
			metalSpheres->SetName("Metal Spheres");
			glassSpheres->SetName("Glass Spheres");
			globalSpheres->SetName("Global Spheres");

			hittableList->Add(diffuseSpheres);
			hittableList->Add(metalSpheres);
			hittableList->Add(glassSpheres);
			hittableList->Add(globalSpheres);
			break;
		}

		case 3:
		{
			TexturePtr noiseTexture = std::make_shared<NoiseTexture>(4.0f);
			MaterialPtr noiseMaterial = std::make_shared<Lambertian>(noiseTexture);
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, -1000.0f, 0.0f), 1000.0f, noiseMaterial));
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, 2.0f, 0.0f), 2.0f, noiseMaterial));

			MaterialPtr lightRect = std::make_shared<DiffuseLight>(Color3(1.0f), 4.0f);
			hittableList->Add(std::make_shared<XyRect>(Mat2x2{ Point2{ 3.0f, 1.0f }, Point2{ 5.0f, 3.0f } }, -2.0f, lightRect));

			break;
		}

		case 4:
		{
			MaterialPtr red = std::make_shared<Lambertian>(Color3(0.65f, 0.05f, 0.05f));
			MaterialPtr white = std::make_shared<Lambertian>(Color3(0.73f, 0.73f, 0.73f));
			MaterialPtr green = std::make_shared<Lambertian>(Color3(0.12f, 0.45f, 0.15f));
			MaterialPtr light = std::make_shared<DiffuseLight>(Color3(15.0f, 15.0f, 15.0f));

			HittableObjectPtr sideLight = std::make_shared<XzRect>(Mat2x2{ Point2{ 213.0f, 227.0f}, Point2{ 343.0f, 332.0f } }, 554.0f, light);
			HittableObjectPtr side1     = std::make_shared<YzRect>(Mat2x2{ Point2{ 0.0f, 0.0f },    Point2{ 555.0f, 555.0f } }, 555.0f, green);
			HittableObjectPtr side2     = std::make_shared<YzRect>(Mat2x2{ Point2{ 0.0f, 0.0f },    Point2{ 555.0f, 555.0f } }, 0.0f, red);
			HittableObjectPtr side3     = std::make_shared<XzRect>(Mat2x2{ Point2{ 0.0f, 0.0f },    Point2{ 555.0f, 555.0f } }, 0.0f, white);
			HittableObjectPtr side4     = std::make_shared<XzRect>(Mat2x2{ Point2{ 0.0f, 0.0f },    Point2{ 555.0f, 555.0f } }, 555.0f, white);
			HittableObjectPtr side5     = std::make_shared<XyRect>(Mat2x2{ Point2{ 0.0f, 0.0f },    Point2{ 555.0f, 555.0f } }, 555.0f, white);

			sideLight->SetName("Side Light");
			side1->SetName("Side 1");
			side2->SetName("Side 2");
			side3->SetName("Side 3");
			side4->SetName("Side 4");
			side5->SetName("Side 5");

			hittableList->Add(sideLight);
			hittableList->Add(side1);
			hittableList->Add(side2);
			hittableList->Add(side3);
			hittableList->Add(side4);
			hittableList->Add(side5);
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, 0.0f, 0.0f), 0.5f, light));

			TexturePtr containterTexture2d = std::make_shared<Texture2D>("Resources/container2.png");
			TexturePtr containterSpecTexture2d = std::make_shared<Texture2D>("Resources/container2_specular.png");

			//TexturePtr noiseTexture = std::make_shared<NoiseTexture>(4.0f);

			MaterialPtr lambMaterial = std::make_shared<Lambertian>(containterTexture2d);
			MaterialPtr metalMaterial = std::make_shared<ShinyMetal>(containterSpecTexture2d);
			lambMaterial->AddMaterial(metalMaterial);
			//hittableList->Add(std::make_shared<Box>(Point3(130.0f, 0.0f, 65.0f), Point3(295.0f, 165.0f, 230.0f), lambMaterial));
			//hittableList->Add(std::make_shared<Box>(Point3(265.0f, 0.0f, 295.0f), Point3(430.0f, 330.0f, 460.0f), white));

			HittableObjectPtr box1 = std::make_shared<Box>(Point3(0.0f, 0.0f, 0.0f), Point3(165.0f, 330.0f, 165.0f), white);
			box1->SetName("Box 1");
			//box1 = std::make_shared<RotateZ>(box1, 10.0f);
			//box1 = std::make_shared<Translate>(box1, Vec3(265.0f, 0.0f, 295.0f));
			//hittableList->Add(box1);

			HittableObjectPtr box2 = std::make_shared<Box>(Point3(0.0f, 0.0f, 0.0f), Point3(165.0f, 165.0f, 165.0f), white);
			box2->SetName("Box 2");
			//box2 = std::make_shared<RotateZ>(box2, -18.0f);
			//box2 = std::make_shared<Translate>(box2, Vec3(130.0f, 0.0f, 65.0f));
			
			//hittableList->Add(box2);

			hittableList->Add(box1);// std::make_shared<ConstantMedium>(box1, 0.01f, Color3(0.0f, 0.0f, 0.0f)));
			hittableList->Add(box2);// std::make_shared<ConstantMedium>(box2, 0.01f, Color3(1.0f, 1.0f, 1.0f)));

			break;
		}

		case 5:
		{
			HittableObjectList boxes1;
			MaterialPtr ground = std::make_shared<Lambertian>(Color3(0.48f, 0.83f, 0.53f));

			const int32_t boxes_per_side = 20;
			for (int32_t i = 0; i < boxes_per_side; i++)
			{
				for (int32_t j = 0; j < boxes_per_side; j++) 
				{
					float w = 100.0f;
					float x0 = -1000.0f + i * w;
					float z0 = -1000.0f + j * w;
					float y0 = 0.0f;
					float x1 = x0 + w;
					float y1 = Utils::Random::RandomFloat(1.0f, 101.0f);
					float z1 = z0 + w;

					HittableObjectPtr box = std::make_shared<Box>(Point3(x0, y0, z0), Point3(x1, y1, z1), ground);

					HittableObjectPtr box_trans = std::make_shared<Translate>(box, Vec3(0.0f));

					std::string name("Box Object ");
					name.append(std::to_string(j + i * boxes_per_side));
					box_trans->SetName(name);

					boxes1.Add(box_trans);
				}
			}

			HittableObjectPtr groundBoxes = std::make_shared<BVHnode>(boxes1, 0.0f, 1.0f);
			groundBoxes->SetName("Ground Boxes");
			hittableList->Add(groundBoxes);

			MaterialPtr light = std::make_shared<DiffuseLight>(Color3(1.0f), 7.0f);
			HittableObjectPtr lightRect = std::make_shared<XzRect>(Mat2x2{ Point2(123.0f, 147.0f), Point2(423.0f, 412.0f) }, 554.0f, light);
			HittableObjectPtr lightTranslate = std::make_shared<Translate>(lightRect, Vec3(0.0f));

			lightRect->SetName("Light Rect");
			lightTranslate->SetName("Light");

			hittableList->Add(lightTranslate);

			Point3 center1(400.0f, 400.0f, 200.0f);
			Point3 center2 = center1 + Point3(30.0f, 0.0f, 0.0f);
			MaterialPtr moving_sphere_material = std::make_shared<Lambertian>(Color3(0.7f, 0.3f, 0.1f));
			hittableList->Add(std::make_shared<MovingSphere>(center1, center2, 0.0f, 1.0f, 50.0f, moving_sphere_material));

			MaterialPtr dielectric = std::make_shared<Dielectric>(1.5f);
			MaterialPtr metal = std::make_shared<Metal>(Color3(0.8f, 0.8f, 0.9f), 0.0f);
			hittableList->Add(std::make_shared<Sphere>(Point3(260.0f, 150.0f, 45.0f), 50.0f, dielectric));
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f, 150.0f, 145.0f), 50.0f, metal));

			HittableObjectPtr boundary = std::make_shared<Sphere>(Point3(360.0f, 150.0f, 145.0f), 70.0f, dielectric);
			hittableList->Add(boundary);
			hittableList->Add(std::make_shared<ConstantMedium>(boundary, 0.2f, Color3(0.2f, 0.4f, 0.9f)));
			boundary = std::make_shared<Sphere>(Point3(0.0f, 0.0f, 0.0f), 5000.0f, dielectric);
			HittableObjectPtr mistMedium = std::make_shared<ConstantMedium>(boundary, 0.0001f, Color3(1.0f, 1.0f, 1.0f));
			mistMedium->SetName("Mist Medium");
			hittableList->Add(mistMedium);

			TexturePtr earthTexture = std::make_shared<Texture2D>("Resources/8081_earthmap10k.jpg");
			MaterialPtr emat = std::make_shared<Lambertian>(earthTexture);
			HittableObjectPtr earth = std::make_shared<Sphere>(Point3(400.0f, 200.0f, 400.0f), 100.0f, emat);
			earth->SetName("Earth");
			hittableList->Add(earth);
			TexturePtr pertext = std::make_shared<NoiseTexture>(0.1f);

			MaterialPtr lambertian = std::make_shared<Lambertian>(pertext);
			hittableList->Add(std::make_shared<Sphere>(Point3(220.0f, 280.0f, 300.0f), 80.0f, lambertian));

			HittableObjectList boxes2;
			MaterialPtr white = std::make_shared<Lambertian>(Color3(0.73f, 0.73f, 0.73f));
			int32_t ns = 1000;
			for (int32_t j = 0; j < ns; j++)
				boxes2.Add(std::make_shared<Sphere>(Utils::Random::RandomVec3(0.0f, 165.0f), 10.0f, white));
			
			HittableObjectPtr hittableBox = std::make_shared<BVHnode>(boxes2, 0.0f, 1.0f);
			HittableObjectPtr hittable = std::make_shared<Rotate>(hittableBox, 0.0f, 0.0f, 0.0f);
			hittableList->Add(std::make_shared<Translate>(hittable, Vec3(-100.0f, 270.0f, 395.0f)));

			break;
		}

		default:
		case 1:
		{
			TexturePtr checker = std::make_shared<CheckerTexture>(Color3(0.2f, 0.3f, 0.1f), Color3(0.9f));
			TexturePtr earthTexture2d = std::make_shared<Texture2D>("Resources/8081_earthmap10k.jpg");
			MaterialPtr sphereMaterial1 = std::make_shared<Lambertian>(earthTexture2d);
			TexturePtr sphereTexture2d = std::make_shared<Texture2D>("Resources/5672_mars_10k_color.jpg");
			MaterialPtr sphereMaterial2 = std::make_shared<Lambertian>(sphereTexture2d);
			hittableList->Add(std::make_shared<Sphere>(Point3(20.0f, 0.0f, 0.0f), 10.0f, sphereMaterial1));
			hittableList->Add(std::make_shared<Sphere>(Point3(0.0f,  10.0f, 0.0f), 10.0f, sphereMaterial2));
			break;
		}
	}
}

#if ENABLE_TEST
#include "ftl/task_counter.h"
#include "ftl/task_scheduler.h"

#include <assert.h>
#include <stdint.h>

struct NumberSubset {
	uint64_t start;
	uint64_t end;

	uint64_t total;
};

void AddNumberSubset(ftl::TaskScheduler* taskScheduler, void* arg) {
	(void)taskScheduler;
	NumberSubset* subset = reinterpret_cast<NumberSubset*>(arg);

	subset->total = 0;

	while (subset->start != subset->end) {
		subset->total += subset->start;
		++subset->start;
	}

	subset->total += subset->end;
}

/**
 * Calculates the value of a triangle number by dividing the additions up into tasks
 *
 * A triangle number is defined as:
 *         Tn = 1 + 2 + 3 + ... + n
 *
 * The code is checked against the numerical solution which is:
 *         Tn = n * (n + 1) / 2
 */
void test()
{
	// Create the task scheduler and bind the main thread to it
	ftl::TaskScheduler taskScheduler;
	taskScheduler.Init();

	// Define the constants to test
	constexpr uint64_t triangleNum = 47593243ULL;
	constexpr uint64_t numAdditionsPerTask = 10000ULL;
	constexpr uint64_t numTasks = (triangleNum + numAdditionsPerTask - 1ULL) / numAdditionsPerTask;

	// Create the tasks
	// FTL allows you to create Tasks on the stack.
	// However, in this case, that would cause a stack overflow
	ftl::Task* tasks = new ftl::Task[numTasks];
	NumberSubset* subsets = new NumberSubset[numTasks];
	uint64_t nextNumber = 1ULL;

	for (uint64_t i = 0ULL; i < numTasks; ++i) {
		NumberSubset* subset = &subsets[i];

		subset->start = nextNumber;
		subset->end = nextNumber + numAdditionsPerTask - 1ULL;
		if (subset->end > triangleNum) {
			subset->end = triangleNum;
		}

		tasks[i] = { AddNumberSubset, subset };

		nextNumber = subset->end + 1;
	}

	// Schedule the tasks
	ftl::TaskCounter counter(&taskScheduler);
	taskScheduler.AddTasks(numTasks, tasks, ftl::TaskPriority::Normal, &counter);

	// FTL creates its own copies of the tasks, so we can safely delete the memory
	delete[] tasks;

	// Wait for the tasks to complete
	taskScheduler.WaitForCounter(&counter);

	// Add the results
	uint64_t result = 0ULL;
	for (uint64_t i = 0; i < numTasks; ++i) {
		result += subsets[i].total;
	}

	// Test
	assert(triangleNum * (triangleNum + 1ULL) / 2ULL == result);

	// Cleanup
	delete[] subsets;

	// The destructor of TaskScheduler will shut down all the worker threads
	// and unbind the main thread
}

#endif