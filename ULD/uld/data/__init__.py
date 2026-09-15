from .tofu import ToFU_DataModule


def create_datamod(
    dataset_config,
    conv_template_config,
    data_mode_config,
    tokenizer=None,
    **kwargs,
):
    print(dataset_config)
    class_name = dataset_config.get("class_name")
    if not isinstance(class_name, str) or class_name.casefold() != "tofu":
        raise ValueError(f"RECAP supports only the ToFU data module, got {class_name!r}")

    return ToFU_DataModule(
        tokenizer=tokenizer,
        conv_template_config=conv_template_config,
        **dataset_config,
        **data_mode_config,
        **kwargs,
    )


__all__ = ["ToFU_DataModule", "create_datamod"]
