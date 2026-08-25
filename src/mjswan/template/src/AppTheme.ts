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
        styles: {
          // mjviser's own box, and the height of a slider row: Mantine's `xs` input is
          // a third taller, and a row is as tall as its tallest child.
          input: {
            height: "1.875em",
            minHeight: "1.875em",
            padding: "0.375em",
            letterSpacing: "-0.5px",
          },
        },
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
    // mjviser's slider: a thin `xs` track with square corners, and a thumb that is a
    // bar — `thumbSize: 0` collapses Mantine's circle so the box below is the shape.
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
          // The range's ends, marked under the track; `index.css` tucks their labels in.
          mark: {
            transform: "scale(2)",
          },
          markLabel: {
            fontSize: "0.6rem",
            transform: "translate(-50%, 0.05rem)",
            textAlign: "center",
          },
        },
      },
    }),
  },
});

export const vars = themeToVars(theme);
