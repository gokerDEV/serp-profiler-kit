import re

class ColorConverter:

    @staticmethod
    def calculate_contrast_ratio(color1, color2):
        """
        Calculates the contrast ratio between two colors.

        Args:
            color1: The first color string (hex, rgb, rgba, oklch, or display-p3).
            color2: The second color string (hex, rgb, rgba, oklch, or display-p3).

        Returns:
            The contrast ratio (a float), or None if either color is invalid.
        """
        if not color1 or not color2:
            return 0.0

        r1, g1, b1, a1 = ColorConverter.parse_color_string(color1)
        r2, g2, b2, a2 = ColorConverter.parse_color_string(color2)

        if r1 is None or g1 is None or b1 is None or a1 is None or r2 is None or g2 is None or b2 is None or a2 is None:
            return 0.0

        if a2 < 1.0:
          r2, g2, b2 = ColorConverter.blend_with_background(r2, g2, b2, a2, 255, 255, 255)
          
        if a1 < 1.0:
          r1, g1, b1 = ColorConverter.blend_with_background(r1, g1, b1, a1, r2, g2, b2)

        l1 = ColorConverter.rgb_to_luminance(r1, g1, b1)
        l2 = ColorConverter.rgb_to_luminance(r2, g2, b2)

        if l1 > l2:
            return (l1 + 0.05) / (l2 + 0.05)
        else:
            return (l2 + 0.05) / (l1 + 0.05)

    @staticmethod
    def rgb_to_luminance(r, g, b):
        """
        Calculates the relative luminance of a color according to WCAG 2.0.
        https://www.w3.org/TR/WCAG20/#relativeluminancedef

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).

        Returns:
            The relative luminance of the color.
        """
        def format_channel(val):
            val /= 255
            if val <= 0.03928:
                return val / 12.92
            else:
                return ((val + 0.055) / 1.055) ** 2.4

        r, g, b = map(format_channel, (r, g, b))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b


    @staticmethod
    def parse_color_string(color_string):
        """Parses a color string and returns RGBA values.  Handles various formats."""
        color_string = color_string.lower()  # Normalize to lowercase for consistency
        parsers = [
            ColorConverter.hex_to_rgba,
            ColorConverter.parse_rgb_string,
            ColorConverter.parse_oklch_string,
            ColorConverter.parse_display_p3_string,
            ColorConverter.parse_rgba_string,  # Add rgba parser
        ]

        for parser in parsers:
            try:
                result = parser(color_string)
                if result:
                    if len(result) == 3:
                        return (*result, 1.0)
                    elif len(result) == 4:
                        return result
                    else:
                        return None, None, None, None
            except Exception as e:
                print(f"Error parsing color string '{color_string}': {e}")
                return None, None, None, None

        return None, None, None, None

    @staticmethod
    def blend_with_background(r, g, b, a, br, bg, bb):
      """
      Blends an RGBA color with white to simulate the effect of transparency over a white background.

      Args:
          r: The red component (0-255).
          g: The green component (0-255).
          b: The blue component (0-255).
          a: The alpha value (0.0-1.0).
          br: The red component of the background color (0-255).
          bg: The green component of the background color (0-255).
          bb: The blue component of the background color (0-255).

      Returns:
          A tuple (r, g, b) representing the blended color.
      """
      return (
          int(r * a + br * (1 - a)),
          int(g * a + bg * (1 - a)),
          int(b * a + bb * (1 - a))
      )
    
    @staticmethod
    def oklch_to_rgba(l, c, h):
        """
        Converts an OKLCH color to RGB.

        Args:
            l: Lightness (0-1).
            c: Chroma (0-0.4).
            h: Hue (0-360).

        Returns:
            A tuple (r, g, b) representing the RGB values (0-255).
        """
        # 1. Convert from polar to rectangular coordinates
        a = c * __import__('math').cos(__import__('math').radians(h))
        b = c * __import__('math').sin(__import__('math').radians(h))

        # 2. Convert from OKLab to linear sRGB
        def from_linear(c):
            if c >= 0.0031308:
                return 1.055 * (c ** (1.0 / 2.4)) - 0.055
            else:
                return 12.92 * c
            
        def to_linear(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            else:
                return c / 12.92
        
        l_ = l + 0.3963377774 * a + 0.2158037573 * b
        m_ = l - 0.1055613458 * a - 0.0638541728 * b
        s_ = l - 0.0894841775 * a - 1.2914855480 * b

        l_ = l_ ** 3
        m_ = m_ ** 3
        s_ = s_ ** 3

        r = from_linear(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_)
        g = from_linear(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_)
        b = from_linear(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
        
        return (
            int(max(0, min(255, round(r * 255)))),
            int(max(0, min(255, round(g * 255)))),
            int(max(0, min(255, round(b * 255))))
        )
    
    @staticmethod
    def parse_oklch_string(oklch_string):
      """
      Parses an OKLCH color string and returns the corresponding RGBA values.

      Args:
          oklch_string: The OKLCH color string (e.g., "oklch(0.5 0.1 180)").

      Returns:
          A tuple (r, g, b, a) with RGB values (0-255) and alpha (1.0), or None if the input is invalid.
      """
      match = re.match(r"oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)", oklch_string)
      if match:
          l, c, h = map(float, match.groups())

          if 0 <= l <= 1 and 0 <= c <= 0.4 and 0 <= h <= 360:
              r, g, b = ColorConverter.oklch_to_rgba(l, c, h)
              return r, g, b, 1.0  # OKLCH doesn't have alpha, so it defaults to 1.0

      return None

    @staticmethod
    def display_p3_to_rgb(r, g, b):
      """
      Converts a color from the Display P3 color space to sRGB.

      Args:
          r: Red component (0.0-1.0).
          g: Green component (0.0-1.0).
          b: Blue component (0.0-1.0).

      Returns:
          A tuple (r, g, b, a) with RGB values (0-255) and alpha (1.0), or None if the input is invalid.
      """
      def from_linear(c):
        if c >= 0.0031308:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055
        else:
            return 12.92 * c

      r_srgb = from_linear(r)
      g_srgb = from_linear(g)
      b_srgb = from_linear(b)

      return (
          int(max(0, min(255, round(r_srgb * 255)))),
          int(max(0, min(255, round(g_srgb * 255)))),
          int(max(0, min(255, round(b_srgb * 255))))
      )

    @staticmethod
    def parse_display_p3_string(display_p3_string):
      """
      Parses a color string in the Display P3 color space and returns the corresponding RGBA values.

      Args:
          display_p3_string: The Display P3 color string (e.g., "color(display-p3 0.2 0.5 0.8)").

      Returns:
          A tuple (r, g, b, a) with RGB values (0-255) and alpha (1.0), or None if the input is invalid.
      """
      match = re.match(r"color\(display-p3\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\)", display_p3_string)
      if match:
          r, g, b = map(float, match.groups())

          if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
              r, g, b = ColorConverter.display_p3_to_rgb(r, g, b)
              return r, g, b, 1.0  # Display P3 doesn't have alpha, so it defaults to 1.0

      return None, None, None, None


    @staticmethod
    def hex_to_rgba(hex_color):
        """
        Converts a hex color string to an RGB tuple.

        Args:
            hex_color: The hex color string (e.g., "#RRGGBB" or "#RGB").

        Returns:
            A tuple (r, g, b) representing the RGB values (0-255), or None if the input is invalid.
        """
        hex_color = hex_color.lstrip('#')

        if len(hex_color) == 3:
            # Expand shorthand form
            hex_color = ''.join(c * 2 for c in hex_color)

        if len(hex_color) != 6:
            return None  # Invalid format

        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return r, g, b, 1.0
        except ValueError:
            return None  # Invalid hex code

    @staticmethod
    def parse_rgb_string(rgb_string):
        """
        Parses an RGB color string (e.g., "rgb(255, 0, 100)" or "rgba(0, 200, 50, 0.5)")

        Args:
            rgb_string: The RGB color string.

        Returns:
            A tuple (r, g, b, a) with RGB values (0-255) and alpha (0.0-1.0), or None if the input is invalid.
            Alpha defaults to 1.0 if not specified in the string.
        """

        match = re.match(r"rgba?\s*\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", rgb_string)
        if match:
            r, g, b = map(int, match.group(1, 2, 3))
            a = float(match.group(4)) if match.group(4) else 1.0

            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0.0 <= a <= 1.0:
                return r, g, b, a
        return None

    @staticmethod
    def parse_rgba_string(rgba_string):
        """Parses an RGBA color string (e.g., "rgba(255, 0, 100, 0.5)")."""
        match = re.match(r"rgba\s*\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", rgba_string)
        if match:
            r, g, b, a = map(float, match.groups())
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0.0 <= a <= 1.0:
                return int(r), int(g), int(b), float(a)
        return None, None, None, None
