"""
Cliente de IA Generativa
"""

import logging
from typing import Optional, Protocol

from ..config.settings import get_genai_config

logger = logging.getLogger(__name__)


class GenAIProvider(Protocol):
    """Protocolo para proveedores de GenAI"""

    def generate(self, prompt: str) -> str:
        """Genera texto a partir de un prompt"""
        ...


class GeminiProvider:
    """Proveedor de Google Gemini"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro", **config):
        """
        Inicializa proveedor de Gemini

        Args:
            api_key: API key de Google
            model_name: Nombre del modelo
            **config: Configuración adicional (temperature, max_tokens, etc.)
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

            self.generation_config = {
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.95),
                "top_k": config.get("top_k", 40),
                "max_output_tokens": config.get("max_tokens", 2048),
            }

            logger.info(f"Gemini provider initialized: {model_name}")

        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install it with: pip install google-generativeai"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {e}")

    def generate(self, prompt: str) -> str:
        """Genera texto con Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class OpenAIProvider:
    """Proveedor de OpenAI (para futuro soporte)"""

    def __init__(self, api_key: str, model_name: str = "gpt-4", **config):
        """
        Inicializa proveedor de OpenAI

        Args:
            api_key: API key de OpenAI
            model_name: Nombre del modelo
            **config: Configuración adicional
        """
        try:
            import openai

            openai.api_key = api_key
            self.model_name = model_name
            self.temperature = config.get("temperature", 0.7)
            self.max_tokens = config.get("max_tokens", 1024)

            logger.info(f"✅ OpenAI provider initialized: {model_name}")

        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install it with: pip install openai"
            )

    def generate(self, prompt: str) -> str:
        """Genera texto con OpenAI"""
        try:
            import openai

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OllamaProvider:
    """Proveedor de Ollama (modelos locales)"""

    def __init__(self, model_name: str = "mistral", host: str = "http://localhost:11434", **config):
        """
        Inicializa proveedor de Ollama

        Args:
            model_name: Nombre del modelo local
            host: URL del servidor Ollama
            **config: Configuración adicional
        """
        try:
            import requests

            self.model_name = model_name
            self.host = host
            self.temperature = config.get("temperature", 0.7)
            self.session = requests.Session()

            # Verificar conexión
            response = self.session.get(f"{self.host}/api/tags")
            response.raise_for_status()

            logger.info(f"✅ Ollama provider initialized: {model_name}")

        except ImportError:
            raise ImportError("requests not installed. Install it with: pip install requests")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama: {e}")

    def generate(self, prompt: str) -> str:
        """Genera texto con Ollama"""
        try:
            import requests

            response = self.session.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                }
            )
            response.raise_for_status()

            return response.json()["response"]

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class GenAIClient:
    """
    Cliente de IA Generativa - Abstracción que soporta múltiples proveedores
    Responsabilidad: Gestionar proveedor de GenAI y generar texto
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            **config
    ):
        """
        Inicializa cliente de GenAI

        Args:
            provider: Proveedor (gemini, openai, ollama) - usa config si es None
            api_key: API key - usa config si es None
            model_name: Nombre del modelo - usa config si es None
            **config: Configuración adicional
        """
        genai_config = get_genai_config()

        self.provider_name = provider or genai_config.provider
        self.api_key = api_key or genai_config.api_key
        self.model_name = model_name or genai_config.model_name

        # Merge config
        self.config = {
            "temperature": config.get("temperature", genai_config.temperature),
            "max_tokens": config.get("max_tokens", genai_config.max_tokens),
            **config
        }

        # Inicializar proveedor
        self.provider = self._initialize_provider()

        logger.info(f"GenAI Client initialized with provider: {self.provider_name}")

    def _initialize_provider(self) -> GenAIProvider:
        """
        Inicializa el proveedor de GenAI según configuración

        Returns:
            Instancia del proveedor

        Raises:
            ValueError: Si el proveedor no es soportado
        """
        provider_map = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider
        }

        provider_class = provider_map.get(self.provider_name.lower())

        if not provider_class:
            raise ValueError(
                f"Unsupported provider: {self.provider_name}. "
                f"Supported providers: {list(provider_map.keys())}"
            )

        # Inicializar según proveedor
        if self.provider_name.lower() == "ollama":
            # Ollama no requiere API key
            return provider_class(
                model_name=self.model_name,
                **self.config
            )
        else:
            # Gemini y OpenAI requieren API key
            if not self.api_key:
                raise ValueError(
                    f"API key required for provider '{self.provider_name}'. "
                    f"Set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
                )

            return provider_class(
                api_key=self.api_key,
                model_name=self.model_name,
                **self.config
            )

    def generate(self, prompt: str) -> str:
        """
        Genera texto a partir de un prompt

        Args:
            prompt: Texto del prompt

        Returns:
            Texto generado

        Raises:
            Exception: Si la generación falla
        """
        try:
            return self.provider.generate(prompt)
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error generating text: {str(e)}"

    def test_connection(self) -> bool:
        """
        Prueba la conexión con el proveedor

        Returns:
            True si la conexión es exitosa
        """
        try:
            response = self.generate("Test: respond with 'OK'")
            logger.info(f"Connection test successful: {response[:50]}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False