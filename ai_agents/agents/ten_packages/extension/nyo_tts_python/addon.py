from ten_runtime import (
    Addon,
    register_addon_as_extension,
    TenEnv,
)


@register_addon_as_extension("nyo_tts_python")
class NYOTTSExtensionAddon(Addon):
    def on_create_instance(self, ten_env: TenEnv, name: str, context) -> None:
        from .extension import NYOTTSExtension

        ten_env.log_info("NYOTTSExtensionAddon on_create_instance")
        ten_env.on_create_instance_done(NYOTTSExtension(name), context)
