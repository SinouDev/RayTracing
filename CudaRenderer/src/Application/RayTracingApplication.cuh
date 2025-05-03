#pragma once

#include "Walnut/Application.h"

#include "Walnut/Image.h"
#include "Walnut/Timer.h"

#include "Core/CudaCamera.cuh"

#include "Utils/Time.h"
#include "Utils/Color.h"

#include "Core/CudaRenderer.cuh"

class Scene;

class RayTracingLayer : public Walnut::Layer
{

public:

	RayTracingLayer();

	virtual void OnUpdate(float ts) override;

	virtual void OnUIRender() override;

	virtual void OnDetach() override;

	void SavePPM(const char* path = "image.ppm");

	void SavePNG(const char* path = "image.png");

	inline bool& GetAlwaysShowDescInObjectList() { return m_AlwaysShowDescInObjectList; }

	inline CudaRenderer& GetCudaRenderer() { return m_CudaRenderer; }

private:

#if 0

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
		switch (object->GetType())
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

		case HittableObjectTypes::TRANSLATE:
		{
			//ImGui::PushID(++id);
			auto translate = object->GetInstance<Translate>();

			bool treeOpen = HandleHittbleObjectView(translate->GetObject()->GetInstance(), ++id, false);

			if (treeOpen)
			{
				ImGui::TextUnformatted(object->GetName());
				DragFloat3("Position", &translate->GetTranslatePosition()[0], id);
				if (popTree)
				{
					ImGui::TreePop();
				}
				treeNodeOpen = showTree;
			}
			//ImGui::PopID();
			break;
		}

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
				if (popTree)
				{
					ImGui::TreePop();
				}
				treeNodeOpen = showTree;
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
			if (showTree || HandleHittbleObjectView(obj, ++id, false))
			{
				ImGui::Text(rotateX->GetName());
				ImGui::Separator();
				SliderAngle("Angle", &rotateX->GetAngle(), id);
				rotateX->Rotate();

				//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);

				//if (treeOpen)
				{
					if (popTree)
					{
						ImGui::TreePop();
					}
					treeNodeOpen = showTree;
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
			if (showTree || HandleHittbleObjectView(obj, ++id, false))
			{
				ImGui::Text(rotateY->GetName());
				ImGui::Separator();
				ImGui::SliderAngle("Angle", &rotateY->GetAngle());
				rotateY->Rotate();

				//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);

				//if (treeOpen)
				{
					if (popTree)
					{
						ImGui::TreePop();
					}
					treeNodeOpen = showTree;
				}
				//ImGui::TreePop();
			}
			//ImGui::PopID();
			break;
		}

		case HittableObjectTypes::ROTATE_Z:
		{
			//ImGui::PushID(++id);
			auto rotateZ = object->GetInstance<RotateZ>();
			auto obj = rotateZ->GetObject()->GetInstance();
			//if (EnableTreeNode(obj->GetName(), &obj->GetHittable(), id, obj->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(obj->GetType())))
			if (showTree || HandleHittbleObjectView(obj, ++id, false))
			{
				ImGui::Text(rotateZ->GetName());
				ImGui::Separator();
				ImGui::SliderAngle("Angle", &rotateZ->GetAngle());
				rotateZ->Rotate();

				//bool treeOpen = HandleHittbleObjectView(obj, ++id, false, false);

				//if (treeOpen)
				{
					if (popTree)
					{
						ImGui::TreePop();
					}
					treeNodeOpen = showTree;
				}
				//ImGui::TreePop();
			}
			//ImGui::PopID();
			break;
		}

		case HittableObjectTypes::BVH_NODE:
		{
			//ImGui::PushID(++id);
			if (EnableTreeNode(object->GetName(), &object->GetHittable(), id, object->IsHittable(), showTree, m_AlwaysShowDescInObjectList, "Object Type: %s", HittableObject::GetTypeName(object->GetType())))
			{
				auto bvhNode = object->GetInstance<BVHnode>();

				HandleBVHnode(bvhNode, id);

				if (popTree)
				{
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
		ImGui::PopID();
		return treeNodeOpen;
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
			DragInt("Size", (int32_t*)&checker->GetSize(), id);
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

#endif

	void Render();

	static void InitImGuiStyle();

private:

	//std::shared_ptr<HittableObjectList> m_hittableList;
	std::unique_ptr<Walnut::Image> m_FinalImage;
	Camera* m_Camera;
	Walnut::Timer m_TotalTimer;

	float m_AspectRatioComponent[2] = { 16.0f, 9.0f };
	CameraComponent::CameraLens m_CameraInit = { 45.0f, 0.1f, 100.0f, m_AspectRatioComponent[0] / m_AspectRatioComponent[1], 0.0f, 10.0f };
	float m_LastRenderTime = 0.0f, m_TotalRenderTime = 0.0f;
	float m_DrawTime = 0.0f;

	//Renderer m_Renderer;

	Scene* m_CudaScene;
	CameraType m_CameraType = Perspective_Camera;

	CudaRenderer m_CudaRenderer;
	bool m_KeepUpdatingPerframe = false, m_RenderOnce = false;
	bool m_SceneChanged = false;

	ImVec2 m_PaddingCenter{ 0.0f, 0.0f };

	SGOL::Color m_OldAmbientLightColorStart{ 0.0f };
	SGOL::Color m_OldAmbientLightColorEnd{ 0.0f };
	SGOL::Color m_OldAmbientLightColor{ 0.0f };

	int32_t m_GlobalIdTracker = 0;
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
	bool m_UniformAmbientLightingColor = true;
	bool m_UniformAmbientLightingColorOld = false;

	bool m_AlwaysShowDescInObjectList = true;
};