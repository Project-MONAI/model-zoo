# -*- coding: utf-8 -*-
"""
File  : usd_shader_applier.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs USD Material Rendering

"""

import gc
import json
import logging
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scripts.usd_design.shader_leather_red import LeatherRed

logger = logging.getLogger("USDShaderApplier")


class USDShaderApplier:
    def __init__(self, usd_file, shader_file):
        self.usd_file = usd_file
        self.shader_file = shader_file
        self.stage = None
        self._shader_info_cache = None

    def _load_shader_info(self):
        if self._shader_info_cache is not None:
            return self._shader_info_cache

        try:
            with open(self.shader_file, "r") as f:
                self._shader_info_cache = json.load(f)
                return self._shader_info_cache
        except Exception as e:
            logger.error(f"Error reading shader file {os.path.basename(self.shader_file)}: {e}")
            return {"materials": []}

    def _set_shader_default_attributes(self, shader):
        ior_preset_options = (
            "ior_acrylic_glass:0|ior_air:1|ior_crystal:2|ior_diamond:3|ior_emerald:4|"
            "ior_ethanol:5|ior_flint_glass:6|ior_glass:7|ior_honey_21p_water:8|"
            "ior_human_eye_aqueous_humor:9|ior_human_eye_cornea:10|"
            "ior_human_eye_lens:11|ior_human_eye_vitreous_humor:12|"
            "ior_human_skin:13|ior_human_hair:14|ior_human_wet_hair:15|"
            "ior_ice:16|ior_milk:17|ior_olive_oil:18|ior_pearl:19|"
            "ior_plastic:20|ior_sapphire:21|ior_soap_bubble:22|"
            "ior_vacuum:23|ior_water_0c:24|ior_water_35c:25|"
            "ior_water_100c:26|ior_custom:99"
        )
        scattering_colors_options = (
            "scattering_colors_apple:0|scattering_colors_chicken:1|scattering_colors_cream:2|"
            "scattering_colors_ketchup:3|scattering_colors_marble:4|scattering_colors_potato:5|"
            "scattering_colors_skim_milk:6|scattering_colors_whole_milk:7|scattering_colors_skin_1:8|"
            "scattering_colors_skin_2:9|scattering_colors_skin_3:10|scattering_colors_skin_4:11|"
            "scattering_colors_custom:12"
        )
        emission_mode_options = "emission_lx:0|emission_nt:1"

        shader.CreateImplementationSourceAttr("sourceAsset")
        shader.SetSourceAsset(Sdf.AssetPath("OmniSurface/OmniSurfaceBase.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniSurfaceBase", "mdl")

        inputs = [
            (
                "coat_affect_color",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "coat_affect_roughness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "coat_anisotropy",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "coat_anisotropy_rotation",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "soft_range": {"min": 0.0, "max": 1.0}}},
            ),
            ("coat_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "coat_ior",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 1.5,
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 1.0, "max": 5.0},
                    }
                },
            ),
            (
                "coat_ior_preset",
                Sdf.ValueTypeNames.Int,
                {
                    "customData": {"default": 99},
                    "sdrMetadata": {"__SDR__enum_value": "ior_custom", "options": ior_preset_options},
                },
            ),
            ("coat_normal", Sdf.ValueTypeNames.Float3, {"customData": {"default": Gf.Vec3f(0, 0, 0)}}),
            (
                "coat_roughness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.1, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "coat_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            ("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "diffuse_reflection_roughness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "diffuse_reflection_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.8, "range": {"min": 0.0, "max": 1.0}}},
            ),
            ("emission_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "emission_intensity",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 1.0, "soft_range": {"min": 0.0, "max": 1000.0}}},
            ),
            (
                "emission_mode",
                Sdf.ValueTypeNames.Int,
                {
                    "customData": {"default": 0},
                    "sdrMetadata": {"__SDR__enum_value": "emission_lx", "options": emission_mode_options},
                },
            ),
            (
                "emission_temperature",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 6500.0, "soft_range": {"min": 0.0, "max": 10000.0}}},
            ),
            ("emission_use_temperature", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            (
                "emission_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            ("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            ("enable_opacity", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            ("enable_specular_transmission", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            ("enable_thin_film", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            ("excludeFromWhiteMode", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            ("geometry_displacement", Sdf.ValueTypeNames.Float3, {"customData": {"default": Gf.Vec3f(0, 0, 0)}}),
            ("geometry_normal", Sdf.ValueTypeNames.Float3, {"customData": {"default": Gf.Vec3f(0, 0, 0)}}),
            (
                "geometry_opacity",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 1.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "geometry_opacity_threshold",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "metalness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "specular_reflection_anisotropy",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "specular_reflection_anisotropy_rotation",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "soft_range": {"min": 0.0, "max": 1.0}}},
            ),
            ("specular_reflection_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "specular_reflection_ior_preset",
                Sdf.ValueTypeNames.Int,
                {
                    "customData": {"default": 99},
                    "sdrMetadata": {"__SDR__enum_value": "ior_custom", "options": ior_preset_options},
                },
            ),
            (
                "specular_reflection_ior",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 1.5, "soft_range": {"min": 1.0, "max": 5.0}}},
            ),
            (
                "specular_reflection_roughness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.2, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "specular_reflection_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 1.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "specular_retro_reflection_color",
                Sdf.ValueTypeNames.Color3f,
                {"customData": {"default": Gf.Vec3f(1, 1, 1)}},
            ),
            (
                "specular_retro_reflection_roughness",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.3, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "specular_retro_reflection_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            ("specular_transmission_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "specular_transmission_dispersion_abbe",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 0.0,
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 0.0, "max": 100.0},
                    }
                },
            ),
            (
                "specular_transmission_scatter_anisotropy",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": -1.0, "max": 1.0}}},
            ),
            (
                "specular_transmission_scattering_color",
                Sdf.ValueTypeNames.Color3f,
                {"customData": {"default": Gf.Vec3f(0, 0, 0)}},
            ),
            (
                "specular_transmission_scattering_depth",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 0.0,
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 0.0, "max": 100.0},
                    }
                },
            ),
            (
                "specular_transmission_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "subsurface_anisotropy",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": -1.0, "max": 1.0}}},
            ),
            (
                "subsurface_scale",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 1.0,
                        "omi": {"kit": {"property": {"usd": {"soft_range_ui": {"min": 0.0, "max": 50.0}}}}},
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 0.0, "max": 10.0},
                    }
                },
            ),
            ("subsurface_scattering_color", Sdf.ValueTypeNames.Color3f, {"customData": {"default": Gf.Vec3f(1, 1, 1)}}),
            (
                "subsurface_scattering_colors_preset",
                Sdf.ValueTypeNames.Int,
                {
                    "customData": {"default": 12},
                    "sdrMetadata": {
                        "__SDR__enum_value": "scattering_colors_custom",
                        "options": scattering_colors_options,
                    },
                },
            ),
            (
                "subsurface_transmission_color",
                Sdf.ValueTypeNames.Color3f,
                {"customData": {"default": Gf.Vec3f(1, 1, 1)}},
            ),
            (
                "subsurface_weight",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "range": {"min": 0.0, "max": 1.0}}},
            ),
            (
                "thin_film_ior",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 1.52,
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 1.0, "max": 3.0},
                    }
                },
            ),
            (
                "thin_film_ior_preset",
                Sdf.ValueTypeNames.Int,
                {
                    "customData": {"default": 99},
                    "sdrMetadata": {"__SDR__enum_value": "ior_custom", "options": ior_preset_options},
                },
            ),
            (
                "thin_film_thickness",
                Sdf.ValueTypeNames.Float,
                {
                    "customData": {
                        "default": 400.0,
                        "range": {"min": 0.0, "max": 3.4028235e38},
                        "soft_range": {"min": 0.0, "max": 2000.0},
                    }
                },
            ),
            ("thin_walled", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
        ]

        for name, type_name, data in inputs:
            inp = shader.CreateInput(name, type_name)
            attr = inp.GetAttr()
            if "customData" in data:
                if data["customData"]["default"] is not None:
                    inp.Set(data["customData"]["default"])

                attr.SetCustomData(data["customData"])
            if "sdrMetadata" in data:
                attr.SetMetadata("sdrMetadata", data["sdrMetadata"])

            if "geometry_normal" == name:
                attr.SetMetadata("hidden", True)

        self.stage.GetRootLayer().Save()

    def mdl_bsdf_shader_creater(self, material, shader_data):
        try:
            shader = UsdShade.Shader.Define(self.stage, shader_data["path"])

            self._set_shader_default_attributes(shader)

            input_specs = [
                # (enter name, value type, datapath, conversion function)
                ("coat_affect_color", Sdf.ValueTypeNames.Float, ["coat_affect_color"]),
                ("coat_roughness", Sdf.ValueTypeNames.Float, ["coat_roughness"]),
                ("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f, ["diffuse_reflection", "color"]),
                ("diffuse_reflection_weight", Sdf.ValueTypeNames.Float, ["diffuse_reflection", "weight"]),
                ("emission_intensity", Sdf.ValueTypeNames.Float, ["emission_intensity"]),
                ("metalness", Sdf.ValueTypeNames.Float, ["metalness"]),
                ("specular_reflection_ior", Sdf.ValueTypeNames.Int, ["specular_reflection", "ior"]),
                ("specular_reflection_roughness", Sdf.ValueTypeNames.Float, ["specular_reflection", "roughness"]),
                ("specular_reflection_weight", Sdf.ValueTypeNames.Float, ["specular_reflection", "weight"]),
                (
                    "specular_retro_reflection_roughness",
                    Sdf.ValueTypeNames.Float,
                    ["specular_reflection", "retro_reflection_roughness"],
                ),
                (
                    "specular_transmission_color",
                    Sdf.ValueTypeNames.Color3f,
                    ["specular_reflection", "transmission_color"],
                ),
                ("subsurface_scale", Sdf.ValueTypeNames.Float, ["subsurface", "scale"]),
                ("subsurface_scattering_color", Sdf.ValueTypeNames.Color3f, ["subsurface", "scattering_color"]),
                ("subsurface_transmission_color", Sdf.ValueTypeNames.Color3f, ["subsurface", "transmission_color"]),
            ]

            for input_name, data_type, data_keys in input_specs:
                try:
                    value = shader_data
                    for key in data_keys:
                        if key not in value:
                            logger.warning(f"Warning: Missing key '{key}' in shader data for input '{input_name}'")
                            continue
                        value = value[key]

                    if data_type == Sdf.ValueTypeNames.Color3f:
                        value = Gf.Vec3f(*value)

                    shader.GetInput(input_name).Set(value)
                except Exception as e:
                    logger.error(f"Error setting input '{input_name}': {e}")

            material.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
                shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            )

            return material
        except Exception as e:
            logger.error(f"Error creating MDL BSDF shader: {e}")
            return material

    def preview_bsdf_shader_creater(self, material, shader_data):
        try:
            shader = UsdShade.Shader.Define(self.stage, shader_data["path"])
            shader.CreateIdAttr("UsdPreviewSurface")

            input_specs = [
                # (Enter name, value type, shader_data key,)
                ("clearcoat", Sdf.ValueTypeNames.Float, "clearcoat"),
                ("clearcoatRoughness", Sdf.ValueTypeNames.Float, "clearcoatRoughness"),
                ("diffuseColor", Sdf.ValueTypeNames.Color3f, "base_color"),
                ("ior", Sdf.ValueTypeNames.Float, "ior"),
                ("metallic", Sdf.ValueTypeNames.Float, "metallic"),
                ("opacity", Sdf.ValueTypeNames.Float, "opacity"),
                ("roughness", Sdf.ValueTypeNames.Float, "roughness"),
                ("specular", Sdf.ValueTypeNames.Float, "specular"),
            ]

            for input_name, value_type, data_key in input_specs:
                try:
                    if data_key not in shader_data:
                        logger.warning(f"Warning: Missing key '{data_key}' in preview shader data")
                        continue

                    value = shader_data[data_key]
                    if value_type == Sdf.ValueTypeNames.Color3f:
                        value = Gf.Vec3f(*value)

                    shader.CreateInput(input_name, value_type).Set(value)
                except Exception as e:
                    logger.error(f"Error setting preview input '{input_name}': {e}")

            # Connect shader to material outputs
            material.CreateOutput("surface", Sdf.ValueTypeNames.Token).ConnectToSource(
                shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            )

            return material
        except Exception as e:
            logger.error(f"Error creating preview BSDF shader: {e}")
            return material

    def create_material(self, material_path, shader_data):
        """Creates and returns a USD material with the specified properties."""
        try:
            # Define material and shader
            prim = self.stage.GetPrimAtPath(Sdf.Path(material_path))
            material = (
                UsdShade.Material.Define(self.stage, material_path) if not prim.IsValid() else UsdShade.Material(prim)
            )

            if "mdl_bsdf" in shader_data:
                material = self.mdl_bsdf_shader_creater(material, shader_data["mdl_bsdf"])

            if "preview_bsdf" in shader_data:
                material = self.preview_bsdf_shader_creater(material, shader_data["preview_bsdf"])

            return material
        except Exception as e:
            logger.error(f"Error creating material at {material_path}: {e}")
            return None

    def _create_material_with_override(self, shader_data, leather_red):
        try:
            material = self.create_material(shader_data["path"], shader_data["bsdf"])
            if material and os.path.basename(shader_data["path"]) == "Leather_red":
                return leather_red.create_material(material)
            return material
        except Exception as e:
            logger.error(f"Error creating material with override: {e}")
            return None

    def apply_materials_to_mesh(self, mesh_paths):
        """Applies material properties to the USD file by creating materials and binding them to prims."""
        try:
            self.stage = Usd.Stage.Open(self.usd_file)
            if not self.stage:
                logger.error(f"Error: Could not open USD file {self.usd_file}")
                return False

            leather_red = LeatherRed(self.stage)

            materials_root = "/root/materials"
            if not self.stage.GetPrimAtPath(materials_root):
                UsdGeom.Scope.Define(self.stage, materials_root)

            material_bindings = {
                os.path.basename(shader_data["path"]): self._create_material_with_override(shader_data, leather_red)
                for shader_data in self._load_shader_info()["materials"]
            }

            coronary_material = material_bindings.get("coronary_artery")
            leather_red_material = material_bindings.get("Leather_red")

            for mesh_path in mesh_paths:
                prim = self.stage.GetPrimAtPath(mesh_path)
                if not prim.IsValid():
                    continue

                organ_name = mesh_path.split("/")[2]
                material = (
                    material_bindings.get(organ_name)
                    or (coronary_material if "coronary_artery" in organ_name else None)
                    or (leather_red_material if organ_name in ["heart", "atrial_appendage_left"] else None)
                )

                if material:
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    if organ_name in ["heart", "atrial_appendage_left"] and material == leather_red_material:
                        leather_red.bind_material(prim, material)
                    else:
                        binding_api.Bind(material)

            # Memory management cleanup
            self._shader_info_cache = None
            material_bindings.clear()
            del leather_red

            # Save the modified USD file
            self.stage.GetRootLayer().Save()
            logger.info(f"Successfully applied materials to {os.path.basename(self.usd_file)}")
        except Exception as e:
            logger.error(f"Error applying materials: {e}")
            return False
        finally:
            if self.stage:
                self.stage = None
            gc.collect()
