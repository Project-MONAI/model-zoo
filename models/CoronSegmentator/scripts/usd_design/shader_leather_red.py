#!/usr/bin/env python

"""
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0
"""
from pxr import Gf, Sdf, UsdShade


class LeatherRed:
    def __init__(self, stage):
        self.stage = stage
        self.base_path = "/root/materials/Leather_red"
        self.baseColor_url = self._get_texture_path("plane_divided_DefaultMaterial_BaseColor.jpg")
        self.roughness_url = self._get_texture_path("plane_divided_DefaultMaterial_Roughness.jpg")
        self.normal_url = self._get_texture_path("plane_divided_DefaultMaterial_Normal.jpg")

    @staticmethod
    def _get_texture_path(filename):
        return f"./textures/{filename}"

    def define_shader(self, path, mdl_source=None, sub_identifier=None):
        shader = UsdShade.Shader.Define(self.stage, path)

        if mdl_source and sub_identifier:
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath(mdl_source), "mdl")
            shader.SetSourceAssetSubIdentifier(sub_identifier, "mdl")
        return shader

    def _create_shader(self, shader_path, shader_name, mdl_source=None, sub_identifier=None):
        shader_path = shader_path.AppendChild(shader_name)
        prim = self.stage.GetPrimAtPath(shader_path)
        if not prim.IsValid():
            return self.define_shader(shader_path, mdl_source, sub_identifier)
        return UsdShade.Shader(prim)

    def _create_node_graph(self, nodegraph_path):
        prim = self.stage.GetPrimAtPath(nodegraph_path)
        if not prim.IsValid():
            return UsdShade.NodeGraph.Define(self.stage, nodegraph_path)
        return UsdShade.NodeGraph(prim)

    def _create_texture_nodegraph(self, nodegraph_name, texture_url):
        """Generic method to create texture node graphs."""
        nodegraph_path = Sdf.Path(self.base_path).AppendChild(nodegraph_name)
        node_graph = UsdShade.NodeGraph.Define(self.stage, nodegraph_path)

        # Common input configuration
        texture = node_graph.CreateInput("texture", Sdf.ValueTypeNames.Asset)
        texture.Set(texture_url)
        texture.GetAttr().SetColorSpace("raw")
        texture.GetAttr().SetCustomData({"default": ""})
        texture.GetAttr().SetMetadata("displayGroup", "Bitmap parameters")
        texture.GetAttr().SetMetadata("displayName", "Bitmap file")

        mono_source = node_graph.CreateInput("mono_source", Sdf.ValueTypeNames.Int)
        mono_source.Set(2)
        mono_source.GetAttr().SetCustomData({"default": 1})
        mono_source.GetAttr().SetMetadata(
            "sdrMetadata",
            {
                "__SDR__enum_value": "mono_average",
                "options": "mono_alpha:0|mono_average:1|mono_luminance:2|mono_maximum:3",
            },
        )
        mono_source.GetAttr().SetMetadata("displayGroup", "Bitmap parameters")
        mono_source.GetAttr().SetMetadata("displayName", "Scalar mode")

        brightness = node_graph.CreateInput("brightness", Sdf.ValueTypeNames.Float)
        brightness.Set(1.0)
        brightness.GetAttr().SetCustomData({"default": 1.0, "soft_range": {"min": 0.0, "max": 1.0}})
        brightness.GetAttr().SetMetadata("displayGroup", "Bitmap parameters")
        brightness.GetAttr().SetMetadata("displayName", "Brightness")

        contrast = node_graph.CreateInput("contrast", Sdf.ValueTypeNames.Float)
        contrast.Set(1.0)
        contrast.GetAttr().SetCustomData({"default": 1.0, "soft_range": {"min": 0.0, "max": 1.0}})
        contrast.GetAttr().SetMetadata("displayGroup", "Bitmap parameters")
        contrast.GetAttr().SetMetadata("displayName", "Contrast")

        scaling = node_graph.CreateInput("scaling", Sdf.ValueTypeNames.Float2)
        scaling.Set(Gf.Vec2f(1.0, 1.0))
        scaling.GetAttr().SetCustomData({"default": Gf.Vec2f(1.0, 1.0)})
        scaling.GetAttr().SetMetadata("displayGroup", "Placement")
        scaling.GetAttr().SetMetadata("displayName", "Tiling")

        translation = node_graph.CreateInput("translation", Sdf.ValueTypeNames.Float2)
        translation.Set(Gf.Vec2f(0.0, 0.0))
        translation.GetAttr().SetCustomData({"default": Gf.Vec2f(0.0, 0.0)})
        translation.GetAttr().SetMetadata("displayGroup", "Placement")
        translation.GetAttr().SetMetadata("displayName", "Offset")

        rotation = node_graph.CreateInput("rotation", Sdf.ValueTypeNames.Float)
        rotation.Set(0.0)
        rotation.GetAttr().SetCustomData({"default": 0.0})
        rotation.GetAttr().SetMetadata("displayGroup", "Placement")
        rotation.GetAttr().SetMetadata("displayName", "Rotation")

        clip = node_graph.CreateInput("clip", Sdf.ValueTypeNames.Bool)
        clip.Set(False)
        clip.GetAttr().SetCustomData({"default": False})
        clip.GetAttr().SetMetadata("displayGroup", "Placement")
        clip.GetAttr().SetMetadata("displayName", "Clip")

        texture_space = node_graph.CreateInput("texture_space", Sdf.ValueTypeNames.Int)
        texture_space.Set(0)
        texture_space.GetAttr().SetCustomData({"default": 0, "range": {"min": 0, "max": 3}})
        texture_space.GetAttr().SetMetadata("displayGroup", "Placement")
        texture_space.GetAttr().SetMetadata("displayName", "UV space index")

        invert = node_graph.CreateInput("invert", Sdf.ValueTypeNames.Bool)
        invert.Set(False)
        invert.GetAttr().SetCustomData({"default": False})
        invert.GetAttr().SetMetadata("displayGroup", "Bitmap parameters")
        invert.GetAttr().SetMetadata("displayName", "Invert image")

        # image_texture = {
        #     "texture": texture, "mono_source": mono_source, "brightness": brightness,
        #     "contrast": contrast, "translation": translation,"scaling": scaling,
        #     "rotation": rotation, "invert": invert, "texture_space": texture_space, "clip": clip,
        # }

        file_texture = self._create_shader(
            nodegraph_path,
            "file_texture",
            "nvidia/core_definitions.mdl",
            "file_texture(texture_2d,::base::mono_mode,float,float,float2,float2,float,bool,int,bool)",
        )

        _texture = file_texture.CreateInput("texture", Sdf.ValueTypeNames.Asset)
        _texture.GetAttr().SetColorSpace("sRGB")
        _texture.ConnectToSource(texture)
        file_texture.CreateInput("mono_source", Sdf.ValueTypeNames.Int).ConnectToSource(mono_source)
        file_texture.CreateInput("brightness", Sdf.ValueTypeNames.Float).ConnectToSource(brightness)
        file_texture.CreateInput("contrast", Sdf.ValueTypeNames.Float).ConnectToSource(contrast)
        file_texture.CreateInput("translation", Sdf.ValueTypeNames.Float2).ConnectToSource(translation)
        file_texture.CreateInput("scaling", Sdf.ValueTypeNames.Float2).ConnectToSource(scaling)
        file_texture.CreateInput("rotation", Sdf.ValueTypeNames.Float).ConnectToSource(rotation)
        file_texture.CreateInput("clip", Sdf.ValueTypeNames.Bool).ConnectToSource(clip)
        file_texture.CreateInput("texture_space", Sdf.ValueTypeNames.Int).ConnectToSource(texture_space)
        file_texture.CreateInput("invert", Sdf.ValueTypeNames.Bool).ConnectToSource(invert)

        file_texture_out = file_texture.CreateOutput("out", Sdf.ValueTypeNames.Token)

        # file_texture = self._create_file_texture(nodegraph_path, image_texture)
        construct_color = self._create_construct_color(nodegraph_path, file_texture_out)
        construct_float = self._create_construct_float(nodegraph_path, file_texture_out)

        # Create output connections
        node_graph.CreateOutput("tex", Sdf.ValueTypeNames.Token).ConnectToSource(file_texture_out)
        node_graph.CreateOutput("color", Sdf.ValueTypeNames.Color3f).ConnectToSource(construct_color)
        node_graph.CreateOutput("mono", Sdf.ValueTypeNames.Float).ConnectToSource(construct_float)

        for channel, axis in [("r", "x"), ("g", "y"), ("b", "z")]:
            node_graph.CreateOutput(channel, Sdf.ValueTypeNames.Float).ConnectToSource(
                self._create_channel_shader(nodegraph_path, axis, construct_color).ConnectableAPI(), "out"
            )

        return node_graph

    def _create_file_texture(self, parent_path, image_texture=None):
        """Creates a file texture shader with standard configuration."""
        file_texture = self._create_shader(
            parent_path,
            "file_texture",
            "nvidia/core_definitions.mdl",
            "file_texture(texture_2d,::base::mono_mode,float,float,float2,float2,float,bool,int,bool)",
        )

        for param in [
            "texture",
            "mono_source",
            "brightness",
            "contrast",
            "translation",
            "scaling",
            "rotation",
            "invert",
            "texture_space",
            "clip",
        ]:
            texture = file_texture.CreateInput(param, image_texture[param].GetTypeName())
            if param == "texture":
                texture.GetAttr().SetColorSpace("sRGB")
            texture.ConnectToSource(image_texture[param])

        return file_texture.CreateOutput("out", Sdf.ValueTypeNames.Token)

    def _create_construct_float(self, parent_path, file_texture):
        """Creates a float constructor shader."""
        construct_float = self._create_shader(
            parent_path, "construct_float", "nvidia/aux_definitions.mdl", "construct_float(::base::texture_return)"
        )

        construct_float.CreateInput("a", Sdf.ValueTypeNames.Token).ConnectToSource(file_texture)
        return construct_float.CreateOutput("out", Sdf.ValueTypeNames.Float)

    def _create_construct_color(self, parent_path, file_texture):
        """Creates a color constructor shader."""
        construct_color = self._create_shader(
            parent_path, "construct_color", "nvidia/aux_definitions.mdl", "construct_color(::base::texture_return)"
        )

        construct_color.CreateInput("a", Sdf.ValueTypeNames.Token).ConnectToSource(file_texture)
        return construct_color.CreateOutput("out", Sdf.ValueTypeNames.Color3f)

    def _create_channel_shader(self, parent_path, axis, construct_color):
        """Creates a channel extractor shader."""
        axis_shader = self._create_shader(parent_path, axis, "nvidia/aux_definitions.mdl", f"{axis}(color)")
        axis_shader.CreateInput("a", Sdf.ValueTypeNames.Color3f).ConnectToSource(construct_color)
        return axis_shader

    def _configure_bsdf(self, bsdf, color_tex, rough_tex, normal_map):
        """Configures the MDL principled BSDF inputs."""
        params = {
            "coat_roughness": 0.03,
            "diffuse_reflection_weight": 1.0,
            "emission_intensity": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "specular_retro_reflection_roughness": 0.5,
            "specular_retro_reflection_weight": 0.0,
            "subsurface_scale": 50.0,
            "subsurface_scattering_color": Gf.Vec3f(1, 0.2, 0.1),
            "subsurface_transmission_color": Gf.Vec3f(0.8, 0.8, 0.8),
        }
        for name, value in params.items():
            bsdf.GetInput(name).Set(value)

        # Create type casts
        color_cast = self._create_shader(
            Sdf.Path(self.base_path),
            "MDL_TypeCast1",
            "nvidia/aux_definitions.mdl",
            "construct_color(::base::texture_return)",
        )
        color_cast.CreateInput("a", Sdf.ValueTypeNames.Token).ConnectToSource(color_tex.ConnectableAPI(), "tex")

        float_cast = self._create_shader(
            Sdf.Path(self.base_path), "MDL_TypeCast2", "nvidia/aux_definitions.mdl", "construct_float"
        )
        float_cast.CreateInput("a", Sdf.ValueTypeNames.Token).ConnectToSource(rough_tex.ConnectableAPI(), "tex")

        # Connect BSDF inputs
        bsdf.GetInput("diffuse_reflection_color").ConnectToSource(color_cast.ConnectableAPI(), "out")
        bsdf.GetInput("specular_reflection_roughness").ConnectToSource(float_cast.ConnectableAPI(), "out")
        bsdf.GetInput("geometry_normal").ConnectToSource(normal_map.ConnectableAPI(), "out")
        bsdf.GetInput("specular_transmission_color").ConnectToSource(color_cast.ConnectableAPI(), "out")

    def _create_normal_map(self):
        """Creates the normal map shader network."""
        shader = self._create_shader(
            Sdf.Path(self.base_path), "MDL_NormalMap", "nvidia/core_definitions.mdl", "normalmap_texture"
        )
        inputs = [
            ("clip", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            (
                "factor",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 1.0, "soft_range": {"min": 0.0, "max": 1.0}}},
            ),
            ("flip", Sdf.ValueTypeNames.Bool, {"customData": {"default": False}}),
            (
                "rotation",
                Sdf.ValueTypeNames.Float,
                {"customData": {"default": 0.0, "soft_range": {"min": 0.0, "max": 360.0}}},
            ),
            ("scaling", Sdf.ValueTypeNames.Float2, {"customData": {"default": Gf.Vec2f(1.0, 1.0)}}),
            ("texture", Sdf.ValueTypeNames.Asset, {"customData": {"default": ""}, "colorSpace": "raw"}),
            (
                "texture_space",
                Sdf.ValueTypeNames.Int,
                {"customData": {"default": 0, "soft_range": {"min": 0, "max": 3}}},
            ),
            ("translation", Sdf.ValueTypeNames.Float2, {"customData": {"default": Gf.Vec2f(0.0, 0.0)}}),
        ]

        for name, type_name, data in inputs:
            inp = shader.CreateInput(name, type_name)
            if data["customData"]["default"]:
                inp.Set(data["customData"]["default"])

            if "customData" in data:
                inp.GetAttr().SetCustomData(data["customData"])
            if "colorSpace" in data:
                inp.GetAttr().SetColorSpace(data["colorSpace"])

        shader.GetInput("texture").Set(self.normal_url)
        shader.CreateOutput("out", Sdf.ValueTypeNames.Float3)

        return shader

    def _create_preview_bsdf(self):
        """Creates the USD Preview Surface network."""
        # bsdf = self._create_shader(Sdf.Path(self.base_path), "preview_Principled_BSDF")
        # bsdf.CreateIdAttr('UsdPreviewSurface')
        bsdf = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.base_path}/preview_Principled_BSDF"))

        # Create texture components
        uv_reader = self._create_shader(Sdf.Path(self.base_path), "preview_uvmap")
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        color_tex = self._create_shader(Sdf.Path(self.base_path), "preview_Image_Texture")
        color_tex.CreateIdAttr("UsdUVTexture")
        color_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(self.baseColor_url)
        color_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        color_tex.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2.0, 2.0))

        rough_tex = self._create_shader(Sdf.Path(self.base_path), "preview_Image_Texture_001")
        rough_tex.CreateIdAttr("UsdUVTexture")
        rough_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(self.roughness_url)
        rough_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        rough_tex.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2.0, 2.0))

        normal_tex = self._create_shader(Sdf.Path(self.base_path), "preview_Image_Texture_002")
        normal_tex.CreateIdAttr("UsdUVTexture")
        normal_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(self.normal_url)
        normal_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")

        # Configure BSDF inputs
        bsdf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(color_tex.ConnectableAPI(), "rgb")
        bsdf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rough_tex.ConnectableAPI(), "r")
        bsdf.CreateInput("normal", Sdf.ValueTypeNames.Float3).ConnectToSource(normal_tex.ConnectableAPI(), "rgb")

        return bsdf

    def create_material(self, material):
        """Main method to create the complete material."""
        # Create MDL networks
        color_tex = self._create_texture_nodegraph("MDL_ImageTexture", self.baseColor_url)
        rough_tex = self._create_texture_nodegraph("MDL_ImageTexture_001", self.roughness_url)
        normal_map = self._create_normal_map()

        bsdf = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.base_path}/MDL_PrincipledBSDF"))
        self._configure_bsdf(bsdf, color_tex, rough_tex, normal_map)

        # Create preview surface
        preview_bsdf = self._create_preview_bsdf()

        # Connect material outputs
        material.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(bsdf.ConnectableAPI(), "out")
        material.CreateOutput("surface", Sdf.ValueTypeNames.Token).ConnectToSource(
            preview_bsdf.ConnectableAPI(), "surface"
        )

        self.stage.GetRootLayer().Save()
        return material

    @staticmethod
    def bind_material(prim, material_path):
        """Binds material to a prim."""
        UsdShade.MaterialBindingAPI(prim).Bind(material_path)
