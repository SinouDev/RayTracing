#pragma once

#include <imgui.h>
#include <cinttypes>
#include <string>

namespace ImGui { namespace Helper {

	static inline void HelpMarkerV(const char* desc, va_list args)
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

	static inline bool TreeNodeV(const char* label, int32_t& id, ImGuiTreeNodeFlags flags, bool condition, bool* changed, bool showTree, bool alwaysShowDesc, const char* descFrm, va_list args)
	{
		if (!condition)
		{
			ImGui::BeginDisabled();
			if (ImGui::TreeNodeEx((void*)(intptr_t)id, flags, label))
				ImGui::TreePop();
			ImGui::EndDisabled();
		}
		bool tree = condition && ImGui::TreeNodeEx((void*)(intptr_t)id, flags, label);
		if (descFrm)
		{
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
		}
		return tree;
	}

	static inline bool DisablableTreeNode(const char* label, bool* check, int32_t& id, bool condition = true, bool* changed = nullptr, bool showTree = true, bool alwaysShowDesc = false, const char* descFrm = nullptr, ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth, ...)
	{
		if (!showTree)
			return true;
		ImGui::PushID(++id);
		bool changedVal = ImGui::Checkbox("", check);
		if (changed)
			*changed = *changed || changedVal;
		ImGui::PopID();
		ImGui::SameLine();
		va_list args;
		va_start(args, label);
		bool res = TreeNodeV(label, id, flags, condition, changed, showTree, alwaysShowDesc, descFrm, args);
		va_end(args);
		return res;
	}

	static inline bool TreeNode(const char* label, int32_t& id, ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth, bool condition = true, bool* changed = nullptr, bool showTree = true, bool alwaysShowDesc = false, const char* descFrm = nullptr, ...)
	{
		va_list args;
		va_start(args, label);
		bool res = TreeNodeV(label, id, flags, condition, changed, showTree, alwaysShowDesc, descFrm, args);
		va_end(args);
		return res;
	}

	static inline bool ColorEdit3(const char* label, float* color, int32_t& id, bool sameLine = true, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (sameLine)
			ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		else ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
		ImGui::PushID(++id);
		ImGui::TextUnformatted(label);
		if (sameLine)
			ImGui::SameLine(); changed = ImGui::ColorEdit3("", color, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderAngle(const char* label, float* deg, int32_t& id, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format == nullptr)
			format = "x: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderAngle("", deg, -360.0f, 360.0f, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderAngle2(const char* label, float* deg, int32_t& id, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderAngle("", deg, -360.0f, 360.0f, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderAngle("", deg + 1, -360.0f, 360.0f, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderAngle3(const char* label, float* deg, int32_t& id, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderAngle("", deg, -360.0f, 360.0f, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderAngle("", deg + 1, -360.0f, 360.0f, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderAngle("", deg + 2, -360.0f, 360.0f, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragFloat(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragFloat("", value, speed, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragFloat2(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragFloat("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragFloat("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragFloat3(const char* label, float* value, int32_t& id, float speed = 1.0f, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragFloat("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragFloat("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragFloat("", value + 2, speed, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragInt(const char* label, int32_t* value, int32_t& id, float speed = 1.0, int32_t min = 0, int32_t max = 0, const char* format = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format == nullptr)
			format = "%.3f";
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragInt("", value, speed, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragInt2(const char* label, int32_t* value, int32_t& id, float speed = 1.0f, int32_t min = 0, int32_t max = 0, const char* format1 = "%d", const char* format2 = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragInt("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragInt("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool DragInt3(const char* label, int32_t* value, int32_t& id, float speed = 1.0f, int32_t min = 0, int32_t max = 0, const char* format1 = "%d", const char* format2 = "%d", const char* format3 = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::DragInt("", value, speed, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragInt("", value + 1, speed, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::DragInt("", value + 2, speed, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderFloat(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderFloat("", value, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderFloat2(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderFloat("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderFloat("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderFloat3(const char* label, float* value, int32_t& id, float min = 0.0f, float max = 0.0f, const char* format1 = nullptr, const char* format2 = nullptr, const char* format3 = nullptr, ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderFloat("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderFloat("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderFloat("", value + 2, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderInt(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format == nullptr)
			format = "%.3f";

		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x));
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderInt("", value, min, max, format, flags);
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderInt2(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format1 = "%d", const char* format2 = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 2.0f)) / 2.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderInt("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderInt("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	static inline bool SliderInt3(const char* label, int32_t* value, int32_t& id, int32_t min = 0, int32_t max = 0, const char* format1 = "%d", const char* format2 = "%d", const char* format3 = "%d", ImGuiBackendFlags flags = 0)
	{
		bool changed;
		if (format1 == nullptr)
			format1 = "x: %.3f";

		if (format2 == nullptr)
			format2 = "y: %.3f";

		if (format3 == nullptr)
			format3 = "z: %.3f";

		ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - (ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x * 3.0f)) / 3.0f);
		ImGui::PushID(++id);
		ImGui::Text(label);
		ImGui::SameLine(); changed = ImGui::SliderInt("", value, min, max, format1, flags);
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderInt("", value + 1, min, max, format2, flags);
		ImGui::PopID();
		ImGui::PushID(++id);
		ImGui::SameLine(); changed = changed || ImGui::SliderInt("", value + 2, min, max, format3, flags);
		ImGui::PopID();
		ImGui::PopID();
		ImGui::PopItemWidth();
		return changed;
	}

	// Copied from imgui_demo.cpp
	// Helper to display a little (?) mark which shows a tooltip when hovered.
	// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
	static inline void HelpMarker(const char* desc, ...)
	{
		va_list args;
		va_start(args, desc);
		HelpMarkerV(desc, args);
		va_end(args);
	}

}}