import argostranslate.package
import argostranslate.translate

import pyarrow as pa
from dora import DoraStatus

from_code = "en"
to_code = "zh"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

# Translate


class Operator:
    """
    Translate content into chinese
    """

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            text = dora_event["value"][0].as_py()
            translatedText = argostranslate.translate.translate(
                text,
                from_code,
                to_code,
            )
            send_output("translatedText", pa.array([translatedText]))
        return DoraStatus.CONTINUE
