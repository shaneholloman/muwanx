import {
  Checkbox,
  ColorInput,
  Select,
  Slider,
  TextInput,
  NumberInput,
  Paper,
  ActionIcon,
  Button,
  createTheme,
  Textarea,
} from "@mantine/core";
import { themeToVars } from "@mantine/vanilla-extract";

export const theme = createTheme({
  fontFamily: "Inter",
  autoContrast: true,
  components: {
    Checkbox: Checkbox.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    ColorInput: ColorInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    Select: Select.extend({
      defaultProps: {
        radius: "sm",
      },
    }),
    Textarea: Textarea.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    TextInput: TextInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    NumberInput: NumberInput.extend({
      defaultProps: {
        radius: "xs",
      },
    }),
    Paper: Paper.extend({
      defaultProps: {
        radius: "xs",
        shadow: "0",
      },
    }),
    ActionIcon: ActionIcon.extend({
      defaultProps: {
        variant: "subtle",
        color: "gray",
        radius: "xs",
      },
    }),
    Button: Button.extend({
      defaultProps: {
        radius: "xs",
        styles: {
          label: {
            fontWeight: 450,
          },
        },
      },
    }),
    // mjviser's slider, prop for prop: a thin `xs` track with square corners and a
    // thumb that is a *bar* rather than a dot — `thumbSize: 0` collapses Mantine's
    // circle so the box below is the whole shape.
    Slider: Slider.extend({
      defaultProps: {
        size: "xs",
        radius: "xs",
        thumbSize: 0,
        styles: {
          root: {
            paddingTop: "0.3em",
            paddingBottom: "0.2em",
          },
          thumb: {
            width: "0.5rem",
            height: "0.75rem",
            background: "var(--slider-color)",
          },
        },
      },
    }),
  },
});

export const vars = themeToVars(theme);
