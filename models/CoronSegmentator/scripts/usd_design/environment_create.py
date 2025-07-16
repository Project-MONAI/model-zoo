# -*- coding: utf-8 -*-
"""
File  : environment_create.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs <brief introduction to USD Environment Rendering>.

"""
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


class EnvironmentSetup:
    def __init__(self, stage):
        self.stage = stage

    def create_dome_light(self):
        """
        Creates a dome light for the scene to illuminate the meshes.
        """
        # Define the light scope
        dome_light = UsdLux.DomeLight.Define(self.stage, "/Environment/Sky")
        prim = dome_light.GetPrim()
        shaping_api = UsdLux.ShapingAPI.Apply(prim)

        # 设置 DomeLight 基础属性
        dome_light.CreateColorTemperatureAttr(6250)
        dome_light.CreateEnableColorTemperatureAttr(True)
        dome_light.CreateExposureAttr(9)
        dome_light.CreateIntensityAttr(1)

        # 设置 ShapingAPI 相关属性
        shaping_api.CreateShapingConeAngleAttr(180)

        # 设置纹理属性
        dome_light.CreateTextureFileAttr(
            Sdf.AssetPath(
                "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Scenes/Templates/Default/SubUSDs/textures/CarLight_512x256.hdr"
            )
        )
        dome_light.CreateTextureFormatAttr("latlong")

        # 设置可见性
        prim.GetAttribute("visibility").Set("inherited")

        # 设置变换操作
        xform = UsdGeom.Xformable(prim)
        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateXYZOp()
        scale_op = xform.AddScaleOp()

        # 设置变换值
        translate_op.Set(Gf.Vec3d(0, 305, 0))
        rotate_op.Set(Gf.Vec3d(0, -90, -90))
        scale_op.Set(Gf.Vec3d(1, 1, 1))

        # 设置操作顺序
        xform.SetXformOpOrder([translate_op, rotate_op, scale_op])

    def create_distant_light(self):
        distant_light = UsdLux.DistantLight.Define(self.stage, "/Environment/DistantLight")
        prim = distant_light.GetPrim()
        shaping_api = UsdLux.ShapingAPI.Apply(prim)

        # Input parameters
        distant_light.CreateColorTemperatureAttr(7250)
        distant_light.CreateEnableColorTemperatureAttr(True)
        distant_light.CreateExposureAttr(10)
        distant_light.CreateIntensityAttr(1)

        # 设置ShapingAPI参数
        shaping_api.CreateShapingConeAngleAttr(180)

        # 设置可见性
        distant_light.GetVisibilityAttr().Set("inherited")

        # 设置变换操作
        xform = UsdGeom.Xformable(prim)
        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateXYZOp()
        scale_op = xform.AddScaleOp()

        # 设置变换值
        translate_op.Set(Gf.Vec3d(0, 305, 0))
        rotate_op.Set(Gf.Vec3d(-105, 0, 0))
        scale_op.Set(Gf.Vec3d(1, 1, 1))

        # 设置操作顺序
        xform.SetXformOpOrder([translate_op, rotate_op, scale_op])

    def create_look_scope(self):
        looks_scope = self.stage.DefinePrim("/Environment/Looks", "Scope")
        # 创建Material
        material = UsdShade.Material.Define(self.stage, "/Environment/Looks/Grid")
        # 创建Shader
        shader = UsdShade.Shader.Define(self.stage, "/Environment/Looks/Grid/Shader")
        shader.CreateIdAttr("Shader")

        # 设置Shader元数据
        shader.CreateImplementationSourceAttr("sourceAsset")
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

        inputs = [
            ("albedo_add", Sdf.ValueTypeNames.Float, 0.0),
            ("albedo_brightness", Sdf.ValueTypeNames.Float, 0.52),
            ("albedo_desaturation", Sdf.ValueTypeNames.Float, 1.0),
            (
                "diffuse_texture",
                Sdf.ValueTypeNames.Asset,
                "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Scenes/Templates/Default/SubUSDs/textures/ov_uv_grids_basecolor_1024.png",
            ),
            ("project_uvw", Sdf.ValueTypeNames.Bool, False),
            ("reflection_roughness_constant", Sdf.ValueTypeNames.Float, 0.333),
            ("texture_rotate", Sdf.ValueTypeNames.Float, 0.0),
            ("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(0.5, 0.5)),
            ("texture_translate", Sdf.ValueTypeNames.Float2, Gf.Vec2f(0, 0)),
            ("world_or_object", Sdf.ValueTypeNames.Bool, False),
        ]

        for name, type_name, value in inputs:
            inp = shader.CreateInput(name, type_name)
            inp.Set(value)
            if name == "diffuse_texture":
                inp.GetAttr().SetColorSpace("sRBG")

        # 创建Shader输出
        output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        # 连接Material输出
        material.CreateOutput("mdl:displacement", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(), "out"
        )
        material.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateOutput("mdl:volume", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")

    def cereate_ground(self):
        mesh = UsdGeom.Mesh.Define(self.stage, "/Environment/ground")
        material_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())

        # 设置几何属性
        mesh.CreateExtentAttr([(-1400, -1400, 0), (1400, 1400, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2])
        mesh.CreatePointsAttr([(-700, -700, 0), (700, -700, 0), (-700, 700, 0), (700, 700, 0)])
        mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        # 设置UV坐标
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
        primvar.Set([(0, 0), (14, 0), (14, 14), (0, 14)])
        primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)

        # 设置其他primvars
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("isMatteObject", Sdf.ValueTypeNames.Bool).Set(False)

        # 材质绑定
        material = UsdShade.Material(self.stage.GetPrimAtPath("/Environment/Looks/Grid"))
        material_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)

        # 设置网格属性
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        mesh.GetVisibilityAttr().Set("inherited")

        # 设置变换操作
        xform = UsdGeom.Xformable(mesh.GetPrim())
        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateXYZOp()
        scale_op = xform.AddScaleOp()

        # 设置变换值
        translate_op.Set(Gf.Vec3d(0, 0, 0))
        rotate_op.Set(Gf.Vec3d(0, -90, -90))
        scale_op.Set(Gf.Vec3d(1, 1, 1))

        # 设置操作顺序
        xform.SetXformOpOrder([translate_op, rotate_op, scale_op])

    def create_ground_collider(self):
        ground_plane = UsdGeom.Plane.Define(self.stage, "/Environment/groundCollider")
        # 设置Plane属性
        # 坐标系设置
        ground_plane.CreateAxisAttr("Y")
        # 显示用途设置
        ground_plane.CreatePurposeAttr("guide")

    def create_environment(self):
        # 创建Environment Xform
        env_xform = UsdGeom.Xform.Define(self.stage, "/Environment")

        # 添加自定义属性
        env_prim = env_xform.GetPrim()
        env_prim.CreateAttribute("ground:size", Sdf.ValueTypeNames.Int).Set(1400)
        env_prim.CreateAttribute("ground:type", Sdf.ValueTypeNames.String).Set("On")

        self.create_dome_light()
        self.create_distant_light()
        self.create_look_scope()
        self.cereate_ground()

        # 保存结果
        self.stage.GetRootLayer().Save()
