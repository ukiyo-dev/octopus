'use client';

import * as React from 'react';
import { motion, isMotionComponent, type HTMLMotionProps } from 'motion/react';
import { cn } from '@/lib/utils';

type AnyProps = Record<string, unknown>;

type DOMMotionProps<T extends HTMLElement = HTMLElement> = Omit<
  HTMLMotionProps<keyof HTMLElementTagNameMap>,
  'ref'
> & { ref?: React.Ref<T> };

type WithAsChild<Base extends object> =
  | (Base & { asChild: true; children: React.ReactElement })
  | (Base & { asChild?: false | undefined });

type SlotProps<T extends HTMLElement = HTMLElement> = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  children?: any;
} & DOMMotionProps<T>;

const INTRINSIC_MOTION_COMPONENTS: Record<string, React.ElementType> = {
  a: motion.a,
  article: motion.article,
  button: motion.button,
  div: motion.div,
  footer: motion.footer,
  h1: motion.h1,
  h2: motion.h2,
  h3: motion.h3,
  header: motion.header,
  img: motion.img,
  input: motion.input,
  label: motion.label,
  li: motion.li,
  main: motion.main,
  nav: motion.nav,
  ol: motion.ol,
  p: motion.p,
  section: motion.section,
  span: motion.span,
  svg: motion.svg,
  textarea: motion.textarea,
  ul: motion.ul,
};

function mergeRefs<T>(
  ...refs: (React.Ref<T> | undefined)[]
): React.RefCallback<T> {
  return (node) => {
    refs.forEach((ref) => {
      if (!ref) return;
      if (typeof ref === 'function') {
        ref(node);
      } else {
        (ref as React.RefObject<T | null>).current = node;
      }
    });
  };
}

function mergeProps<T extends HTMLElement>(
  childProps: AnyProps,
  slotProps: DOMMotionProps<T>,
): AnyProps {
  const merged: AnyProps = { ...childProps, ...slotProps };

  if (childProps.className || slotProps.className) {
    merged.className = cn(
      childProps.className as string,
      slotProps.className as string,
    );
  }

  if (childProps.style || slotProps.style) {
    merged.style = {
      ...(childProps.style as React.CSSProperties),
      ...(slotProps.style as React.CSSProperties),
    };
  }

  return merged;
}

function cloneWithMergedRef<T extends HTMLElement>(
  element: React.ReactElement,
  props: AnyProps,
  ref: React.Ref<T>,
) {
  return React.cloneElement(
    element as React.ReactElement<AnyProps & React.RefAttributes<T>>,
    {
      ...(props as AnyProps & React.RefAttributes<T>),
      ref,
    },
  );
}

function Slot<T extends HTMLElement = HTMLElement>({
  children,
  ref,
  ...props
}: SlotProps<T>) {
  if (!React.isValidElement(children)) return null;

  const isAlreadyMotion =
    typeof children.type === 'object' &&
    children.type !== null &&
    isMotionComponent(children.type);

  const { ref: childRef, ...childProps } = children.props as AnyProps;
  const mergedProps = mergeProps(childProps, props);
  const mergedRef = mergeRefs(childRef as React.Ref<T>, ref);

  if (isAlreadyMotion) {
    // Forwarding refs through cloneElement is intentional here; the lint rule
    // treats it as a render-time ref access even though no ref value is read.
    // eslint-disable-next-line react-hooks/refs
    return cloneWithMergedRef(children, mergedProps, mergedRef);
  }

  if (typeof children.type !== 'string') {
    // eslint-disable-next-line react-hooks/refs
    return cloneWithMergedRef(children, mergedProps, mergedRef);
  }

  const Base = isAlreadyMotion
    ? (children.type as React.ElementType)
    : INTRINSIC_MOTION_COMPONENTS[children.type as keyof typeof INTRINSIC_MOTION_COMPONENTS];

  if (!Base) {
    // eslint-disable-next-line react-hooks/refs
    return cloneWithMergedRef(children, mergedProps, mergedRef);
  }

  return <Base {...mergedProps} ref={mergedRef} />;
}

export {
  Slot,
  type SlotProps,
  type WithAsChild,
  type DOMMotionProps,
  type AnyProps,
};
